from flask import Flask, request, render_template, session
import numpy as np
from logic.hopfield import HopfieldNetwork

app = Flask(__name__)
app.secret_key = 'my-super-secret-key'


@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        # Check for manual vectors to train the network
        if 'manual_vectors' in request.form:
            print(request.form)
            try:
                # Retrieve polarity setting (default to 'bipolar')
                polarity = request.form.get('polarity', 'bipolar')
                input_method = request.form.get('input_method', 'manual')
                hn.mode = polarity
                session['polarity'] = polarity
                session['input_method'] = input_method
                # Check if the input method is on
                if input_method == 'on':
                    # Process the file upload
                    file = request.files['file_vectors']
                    if 'file_vectors' not in request.files:
                        message = "Error: No file part"
                    else:
                        file = request.files['file_vectors']
                        if file.filename == '':
                            message = "Error: No selected file"
                        else:
                            # Read the file and parse the content into an array
                            data = np.genfromtxt(file, delimiter=',', dtype=int)
                            if data.size == 0:
                                message = "Error: No valid input vectors provided"
                            else:
                                hn.train(data)
                                message = "Network trained with the uploaded file."
                    return render_template('index.html', message=message, polarity=session.get('polarity', 'bipolar'), input_method=session.get('input_method', 'manual'))
                # Split the input by lines and parse each line into an array
                input_vectors = request.form['manual_vectors'].strip(
                ).splitlines()
                data = np.array([np.fromstring(line, sep=',', dtype=int)
                                for line in input_vectors])
                if data.size == 0:
                    message = "Error: No valid input vectors provided"
                else:
                    hn.train(data)
                    message = "Network trained with the manual input."
            except ValueError as ve:
                message = f"An error occurred while parsing manual input: {
                    str(ve)}"
        # Check for pattern to test
        if 'pattern' in request.form:
            try:
                pattern = np.fromstring(
                    request.form['pattern'], sep=',', dtype=int)
                output, iterations, info = hn.predict(pattern)
                message = f"Output: {output}, Iterations: {
                    iterations}, Info: {info}"
            except ValueError as ve:
                message = f"An error occurred while parsing the test pattern: {
                    str(ve)}"

    return render_template('index.html', message=message, polarity=session.get('polarity', 'bipolar'), input_method=session.get('input_method', 'manual'))


if __name__ == '__main__':
    hn = HopfieldNetwork()
    app.run(debug=True)
