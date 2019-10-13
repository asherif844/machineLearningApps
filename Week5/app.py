from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/send', methods= ['GET', 'POST'])
def send():
    if request.method == 'POST':
        age2 = request.form['age']
        gender2 = request.form['gender']
        return render_template('age.html', jamaica=age2, gender = gender2)
    return render_template('index.html')


@app.route('/')
def homepage():
    return render_template('homepage.html')

if __name__ == '__main__':
    app.run(debug=True)