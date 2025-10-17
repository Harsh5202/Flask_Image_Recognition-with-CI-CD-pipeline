"""
Flask Image Recognition Application
This module provides a web application for hand sign digit recognition using a deep learning model.
"""

# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """
    Render the main index page with image upload form.

    Returns:
        str: Rendered HTML template for the home page
    """
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Handle image upload and prediction.

    Processes the uploaded image file, runs it through the model,
    and returns the prediction result.

    Returns:
        str: Rendered HTML template with prediction result or error message
    """
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

    except (KeyError, ValueError, IOError, OSError) as error:
        error_message = "File cannot be processed."
        print(f"Error processing file: {error}")
        return render_template("result.html", err=error_message)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
