from flask import Flask, jsonify, request
from chatbot import Chatbot

app = Flask(__name__)

@app.route("/", methods=["GET"])
def get_response():
    user_input = request.form
    return jsonify(chatbot_instance.getResponse(user_input))

if __name__ == "__main__":
    chatbot_instance = Chatbot()
    app.run(debug=True)