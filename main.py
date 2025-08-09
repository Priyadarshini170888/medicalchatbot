from flask import Flask, request, jsonify, render_template
from project_app.utils import get_chat_response
from project_app.utils import retrieve_context


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=['POST'])
def chatbot():
    user_input = request.json.get("input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400  
    
    context = retrieve_context(user_input)    

    augmented_prompt = f"""
    You are a helpful assistant related to HR Query. You can answer questions about company policies, benefits, and other HR-related topics.

    Use the following context to answer the user's question.

    Context:
    {context}

    Question:
    {user_input}

    If the answer is not in the context, say "I do not authorize to answer this. Please Refer Administration."
    """

    response = get_chat_response(augmented_prompt)
    return jsonify({"response": response})


if __name__ == "__main__":  
    app.run(debug=True, host="0.0.0.0", port=5000)



