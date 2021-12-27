from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Congratulations, it's a web app!"

@app.route("/hi")
def who():
    return "Congratulations, this is another way to show HIII!!!!!"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)