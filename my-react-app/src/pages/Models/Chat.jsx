import React, { useState } from "react";
import axios from "axios";

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/chat", {
        prompt: input,
      });

      const botMsg = { role: "bot", text: res.data.response };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        role: "bot",
        text: "⚠️ Error getting response. Check console.",
      };
      setMessages((prev) => [...prev, errorMsg]);
      console.error("Chat error:", err);
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold text-center mb-4 text-blue-700">💬 Mistral Chat</h1>

      <div className="flex-1 overflow-y-auto p-4 rounded-lg bg-white shadow-inner mb-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`my-2 p-3 rounded-xl max-w-[75%] ${
              msg.role === "user"
                ? "ml-auto bg-blue-500 text-white text-right"
                : "mr-auto bg-gray-200 text-gray-800"
            }`}
          >
            <span className="block font-semibold">
              {msg.role === "user" ? "You" : "Bot"}
            </span>
            <p>{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="mr-auto bg-gray-300 text-gray-800 p-3 rounded-xl max-w-[75%] animate-pulse">
            Bot is typing...
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 px-4 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;
