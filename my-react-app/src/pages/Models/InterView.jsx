// import React, { useState } from "react";
// import axios from "axios";

// const API_BASE = "http://localhost:5000"; // Update if backend is hosted elsewhere

// export default function Interview() {
//   const [topic, setTopic] = useState("");
//   const [qaPairs, setQaPairs] = useState([]);
//   const [userAnswer, setUserAnswer] = useState("");
//   const [selectedQA, setSelectedQA] = useState(null);
//   const [confidence, setConfidence] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const fetchInterview = async () => {
//     try {
//       setLoading(true);
//       const res = await axios.post(`${API_BASE}/interview`, { topic });
//       setQaPairs(res.data.qa_pairs);
//       setSelectedQA(null);
//       setConfidence(null);
//       setUserAnswer("");
//     } catch (err) {
//       console.error(err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const evaluate = async () => {
//     if (!selectedQA || !userAnswer) return;
//     try {
//       const res = await axios.post(`${API_BASE}/evaluate`, {
//         question: selectedQA.question,
//         expected_answer: selectedQA.answer,
//         user_answer: userAnswer,
//       });
//       setConfidence(res.data.confidence_score);
//     } catch (err) {
//       console.error(err);
//     }
//   };

//   return (
//     <div className="max-w-4xl mx-auto p-6">
//       <h1 className="text-3xl font-bold mb-6 text-center">🧠 AI Interview Confidence Bot</h1>

//       <div className="mb-6">
//         <input
//           type="text"
//           placeholder="Enter topic (e.g., Software Engineering)"
//           className="w-full px-4 py-2 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
//           value={topic}
//           onChange={(e) => setTopic(e.target.value)}
//         />
//         <button
//           onClick={fetchInterview}
//           className="mt-3 px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
//           disabled={!topic || loading}
//         >
//           {loading ? "Generating..." : "Generate Interview"}
//         </button>
//       </div>

//       <div className="space-y-4">
//         {qaPairs.map((qa, index) => (
//           <div
//             key={index}
//             className={`p-4 border rounded-lg cursor-pointer hover:bg-gray-50 ${
//               selectedQA?.question === qa.question ? "bg-blue-50 border-blue-500" : ""
//             }`}
//             onClick={() => {
//               setSelectedQA(qa);
//               setUserAnswer("");
//               setConfidence(null);
//             }}
//           >
//             <p className="font-semibold">Q{index + 1}: {qa.question}</p>
//             <p className="text-sm italic text-gray-600">Expected: {qa.answer}</p>
//           </div>
//         ))}
//       </div>

//       {selectedQA && (
//         <div className="mt-8">
//           <h2 className="text-xl font-bold mb-2">Your Answer</h2>
//           <p className="mb-2 font-medium text-gray-700">{selectedQA.question}</p>
//           <textarea
//             className="w-full h-32 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
//             placeholder="Type your answer here..."
//             value={userAnswer}
//             onChange={(e) => setUserAnswer(e.target.value)}
//           />
//           <button
//             onClick={evaluate}
//             className="mt-3 px-5 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
//           >
//             Evaluate Confidence
//           </button>

//           {confidence !== null && (
//             <p className="mt-4 text-lg font-semibold text-blue-700">
//               💡 Confidence Score: <span className="text-black">{confidence}</span>
//             </p>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }


import React, { useState } from "react";
import axios from "axios";



const API_BASE = "http://localhost:5000";

export default function Interview() {
  const [topic, setTopic] = useState("");
  const [qaPairs, setQaPairs] = useState([]);
  const [userAnswer, setUserAnswer] = useState("");
  const [selectedQA, setSelectedQA] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [transcriptText, setTranscriptText] = useState("");

  const fetchInterview = async () => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/interview`, { topic });
      console.log("Interview API response:", res.data);

      if (Array.isArray(res.data.qa_pairs)) {
        setQaPairs(res.data.qa_pairs);
      } else {
        console.error("Invalid response format:", res.data);
        alert("Error: Could not load questions.");
        setQaPairs([]);
      }

      setSelectedQA(null);
      setConfidence(null);
      setUserAnswer("");
    } catch (err) {
      console.error("⚠️ Error during fetchInterview:", err);
      alert("Failed to fetch questions.");
    } finally {
      setLoading(false);
    }
  };

  const evaluate = async () => {
    if (!selectedQA || !userAnswer) return;
    try {
      const res = await axios.post(`${API_BASE}/evaluate`, {
        question: selectedQA.question,
        expected_answer: selectedQA.answer,
        user_answer: userAnswer,
      });
      setConfidence(res.data.confidence_score);
    } catch (err) {
      console.error(err);
    }
  };
  
  const fetchTranscriptText = async () => {
  try {
    const res = await axios.get(`${API_BASE}/transcripts-text`);
    setTranscriptText(res.data.transcript || "");
     setUserAnswer(res.data.transcript || "");
  } catch (err) {
    console.error("Failed to fetch transcript text:", err);
  }
};



  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center"> AI Interview Confidence Evaluator</h1>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Enter topic (e.g., Software Engineering)"
          className="w-full px-4 py-2 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        <button
          onClick={fetchInterview}
          className="mt-3 px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
          disabled={!topic || loading}
        >
          {loading ? "Generating..." : "Generate Interview"}
        </button>
      </div>

      <div className="space-y-4">
        {Array.isArray(qaPairs) && qaPairs.length > 0 && qaPairs.map((qa, index) => (
          <div
            key={index}
            className={`p-4 border rounded-lg cursor-pointer hover:bg-gray-50 ${
              selectedQA?.question === qa.question ? "bg-blue-50 border-blue-500" : ""
            }`}
            onClick={() => {
              setSelectedQA(qa);
              setUserAnswer("");
              setConfidence(null);
            }}
          >
            <p className="font-semibold">Q{index + 1}: {qa.question}</p>
            <p className="text-sm italic text-gray-600">Expected: {qa.answer}</p>
          </div>
        ))}
      </div>

      {selectedQA && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-2">Your Answer</h2>
          <p className="mb-2 font-medium text-gray-700">{selectedQA.question}</p>
          <textarea
            className="w-full h-32 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your answer here..."
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
          />
          
          <button
            onClick={evaluate}
            className="mt-3 px-5 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
          >
            Evaluate Confidence
          </button>
          
          <button
            onClick={fetchTranscriptText}
              className="mt-3 px-5 mx-2 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
            >
            Load Transcript
            </button><br /><br />
<button
  onClick={() => fetch("http://localhost:5000/start")}
  className="bg-blue-500 text-white px-4 py-2 rounded"
>
  Start Live Quiz
</button>




          {confidence !== null && (
            <p className="mt-4 text-lg font-semibold text-blue-700">
              💡 Confidence Score: <span className="text-black">{confidence}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
