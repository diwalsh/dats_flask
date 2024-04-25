import os
import zipfile
import base64
import cv2
import tempfile
import torch
import re
from flask import Flask, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash

from cs50 import SQL
from helpers import apology, login_required, normalize, resize, get_patient_images, generate_bouncing_gif
from model import inference, load_model



# configure app
app = Flask(__name__)

# configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# use CS50 built in Library to connect to database via sqlite
# (handles messy sqlachemy connection openings/closures for you)
db = SQL("sqlite:///dats.db")

# Directory to save the generated renderings, etc. 
# this will be fleshed out later 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ensure the upload folder exists!! lol
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# goodbye cache, hello css 
@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    # User reached route via POST (as by submitting a form)
    if request.method == "POST":
        file = request.files['data_zip_file']
        file_like_object = file.stream._file  
        zipfile_ob = zipfile.ZipFile(file_like_object)
        file_names = zipfile_ob.namelist()
        # Filter names to only include the filetype that you want:
        file_names = [file_name for file_name in file_names if file_name.endswith(".png")]
        files = [] 
        for name in file_names:
            if not name.startswith("__MACOSX"):
                img_data = zipfile_ob.open(name).read()
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(img_data)
                    temp_file_path = temp_file.name
                # the below img's need to be sent to a resizing function
                # to be fed into a model for segmentation     
                img = normalize(temp_file_path)
                # from here on, the normalized imgs are prepared for showing on screen to user
                img_data_base64 = cv2.imencode('.png', img)[1].tostring()
                img_data_base64 = base64.b64encode(img_data_base64).decode('utf-8')
                # grabbing relevant data from file name to format for user display
                parts = name.split("/")[-1].split("_")
                case_number = ''.join(filter(str.isdigit, parts[0]))
                day_number = ''.join(filter(str.isdigit, parts[1]))
                slice_number = int(parts[3])
                formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)
                # Create a unique filename for the image
                user_id = str(session["user_id"])
                # pop off any prefixed folder names
                file_name = name.split("/")[-1]
                # Remove the .png suffix from the filename
                file_name = file_name.split(".png")[0]
                img_filename = "{}_{}.png".format(file_name, user_id)
                # create the normalized folder if not already there
                normalized_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'normalized')
                if not os.path.exists(normalized_folder):
                    os.makedirs(normalized_folder)
                img_path = os.path.join(normalized_folder, img_filename)
                # Save the image to the normalized folder
                cv2.imwrite(img_path, img)
                # Append printable image, formatted name, and original name for display use
                files.append((img_data_base64, formatted_name, img_path))
                os.unlink(temp_file_path)  # Delete the temporary file after use
        # Sort files based on the original file name
        files = sorted(files, key=lambda x: x[2])
        session["uploaded_files"] = files
        # Retrieve image paths
        folder_path = "static/uploads/normalized"
        image_paths = get_patient_images(case_number, day_number, folder_path, user_id)
        # Store image paths in session
        session['image_paths'] = image_paths
        return render_template("pngs.html", files=files)
    else:
        return render_template("upload.html")
    
# @app.route("/confirm")
# @login_required
# def confirm():
#     normalized_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'normalized')
#     resized_images = []

#     # Iterate over the files in the normalized folder
#     for filename in os.listdir(normalized_folder):
#         if filename.endswith(".png"):
#             # Read the normalized image
#             normalized_file_path = os.path.join(normalized_folder, filename)
#             img = cv2.imread(normalized_file_path)

#             # Apply the resize function to the normalized image
#             resized_img = resize(img)

#             # Encode the resized image to base64 for display
#             img_data_base64 = cv2.imencode('.png', resized_img)[1].tostring()
#             img_data_base64 = base64.b64encode(img_data_base64).decode('utf-8')

#             # Extract formatted name from filename
#             parts = filename.split("_")
#             case_number = ''.join(filter(str.isdigit, parts[0]))
#             day_number = ''.join(filter(str.isdigit, parts[1]))
#             slice_number = int(parts[3].split(".")[0])
#             formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)

#             # Append the base64 encoded image data, formatted name, and original name to the list
#             resized_images.append((img_data_base64, formatted_name, filename))

#     # Sort files based on the original file name
#     resized_images = sorted(resized_images, key=lambda x: x[2])

#     # Render the confirm template with the resized images
#     return render_template("confirm.html", files=resized_images)

@app.route("/model")
@login_required
def model():
    # retrieve image paths from session, declare device, checkpoint and model
    image_paths = session.get('image_paths', [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "static/ckpt_009.ckpt"
    # load the model
    model = load_model(checkpoint_path)
    # predict!!
    predictions = inference(model, image_paths, device=device)
   
    return render_template("model.html", predictions=predictions, image_paths=image_paths)

@app.route("/render")
@login_required
def render():
    # retrieve image paths from session, declare device, checkpoint and model
    image_paths = session.get('image_paths', [])
    
    # Define generate_title function
    def generate_title(image_path):
        parts = image_path.split("/")[-1].split("_")
        case_number = ''.join(filter(str.isdigit, parts[0]))
        day_number = ''.join(filter(str.isdigit, parts[1]))
        slice_number = int(parts[3].split(".")[0])
        formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)
        return formatted_name
    
    # Extract common prefix from image file names, and user number
    prefix_match = re.match(r'^case(\d+)_day(\d+)_', os.path.basename(image_paths[0]))
    if prefix_match:
        n, o = prefix_match.groups()
    else:
        # Default values if no match found
        n, o = "unknown", "unknown"
    p = session.get('user_number', 'unknown')  # Get the user number from session
    output_gif_path = f'static/gifs/case{n}_day{o}_{p}.gif'
    
    # Generate bouncing GIF from images
    generate_bouncing_gif(image_paths, output_gif_path)
    
    return render_template("render.html", generated_gif=output_gif_path, image_paths=image_paths, generate_title=generate_title)

@app.route("/archive")
@login_required
def archive():
    user_id = session["user_id"]
    renderings = db.execute("SELECT * FROM renderings WHERE user_id = ? ORDER BY created_at DESC", user_id)
    return render_template("archive.html", renderings=renderings)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must create username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must create password", 400)

        # Ensure password confirmation was submitted
        elif not request.form.get("confirmation"):
            return apology("must repeat password", 400)

        # Ensure passwords match!
        elif request.form.get("password") != request.form.get("confirmation"):
            return apology("passwords must match!", 400)

        # Query database for username
        exists = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        # Ensure username doesn't exist
        if len(exists) > 0:
            return apology("username already taken :( ", 400)

        # add user to database
        username = request.form.get("username")
        password = generate_password_hash(request.form.get("password"))
        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, password)

        # Remember which user has logged in
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")
    
    
    
    

