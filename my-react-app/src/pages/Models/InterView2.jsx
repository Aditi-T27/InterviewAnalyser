// import React, { useState } from "react";
// import axios from "axios";

// const API_BASE = "http://localhost:5000";

// export default function InterView2() {
//   const [topic, setTopic] = useState("");
//   const [questions, setQuestions] = useState([]);
//   const [selectedQ, setSelectedQ] = useState(null);
//   const [userAnswer, setUserAnswer] = useState("");
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const fetchInterview = async () => {
//     try {
//       setLoading(true);
//       const res = await axios.post(`${API_BASE}/interview`, { topic });
//       setQuestions(res.data.questions);
//       setSelectedQ(null);
//       setUserAnswer("");
//       setResult(null);
//     } catch (err) {
//       console.error(err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const evaluate = async () => {
//     if (!selectedQ || !userAnswer) return;
//     try {
//       const res = await axios.post(`${API_BASE}/evaluate`, {
//         question: selectedQ,
//         user_answer: userAnswer,
//       });
//       setResult(res.data);
//     } catch (err) {
//       console.error(err);
//     }
//   };

//   return (
//     <div className="max-w-3xl mx-auto p-6">
//       <h1 className="text-2xl font-bold text-center mb-6">🧠 AI Interview Practice</h1>

//       <div className="mb-6">
//         <input
//           type="text"
//           placeholder="Enter topic (e.g., React)"
//           className="w-full px-4 py-2 border rounded-lg"
//           value={topic}
//           onChange={(e) => setTopic(e.target.value)}
//         />
//         <button
//           onClick={fetchInterview}
//           disabled={!topic || loading}
//           className="mt-2 px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
//         >
//           {loading ? "Loading..." : "Generate Questions"}
//         </button>
//       </div>

//       {questions.map((q, i) => (
//         <div
//           key={i}
//           onClick={() => {
//             setSelectedQ(q);
//             setUserAnswer("");
//             setResult(null);
//           }}
//           className={`p-3 border rounded mb-2 cursor-pointer ${
//             selectedQ === q ? "bg-blue-100" : "hover:bg-gray-50"
//           }`}
//         >
//           Q{i + 1}: {q}
//         </div>
//       ))}

//       {selectedQ && (
//         <div className="mt-6">
//           <h2 className="font-semibold mb-1">{selectedQ}</h2>
//           <textarea
//             className="w-full h-28 p-2 border rounded"
//             placeholder="Type your answer..."
//             value={userAnswer}
//             onChange={(e) => setUserAnswer(e.target.value)}
//           />
//           <button
//             onClick={evaluate}
//             className="mt-2 px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700"
//           >
//             Submit Answer
//           </button>

//           {result && (
//             <div className="mt-4 bg-gray-100 p-4 rounded">
//               <p>
//                 💡 <strong>Confidence Score:</strong> {result.confidence_score}
//               </p>
//               <p className="mt-2 text-sm text-gray-700">
//                 ✅ <strong>Model Answer:</strong> {result.expected_answer}
//               </p>
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }


import React, { useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:5000";

export default function InterView2() {
  const [topic, setTopic] = useState("");
  const [questions, setQuestions] = useState([]);
  const [selectedQ, setSelectedQ] = useState(null);
  const [userAnswer, setUserAnswer] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchInterview = async () => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/interview`, { topic });
      // Ensure the response always sets an array
      const safeQuestions = Array.isArray(res.data.questions) ? res.data.questions : [];
      setQuestions(safeQuestions);
      setSelectedQ(null);
      setUserAnswer("");
      setResult(null);
    } catch (err) {
      console.error("Error fetching interview questions:", err);
      setQuestions([]); // fallback
    } finally {
      setLoading(false);
    }
  };

  const evaluate = async () => {
    if (!selectedQ || !userAnswer) return;
    try {
      const res = await axios.post(`${API_BASE}/evaluate`, {
        question: selectedQ,
        user_answer: userAnswer,
      });
      setResult(res.data);
    } catch (err) {
      console.error("Error evaluating answer:", err);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold text-center mb-6">🧠 AI Interview Practice</h1>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Enter topic (e.g., React)"
          className="w-full px-4 py-2 border rounded-lg"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        <button
          onClick={fetchInterview}
          disabled={!topic || loading}
          className="mt-2 px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {loading ? "Loading..." : "Generate Questions"}
        </button>
      </div>

      {Array.isArray(questions) && questions.map((q, i) => (
        <div
          key={i}
          onClick={() => {
            setSelectedQ(q);
            setUserAnswer("");
            setResult(null);
          }}
          className={`p-3 border rounded mb-2 cursor-pointer ${
            selectedQ === q ? "bg-blue-100" : "hover:bg-gray-50"
          }`}
        >
          Q{i + 1}: {q}
        </div>
      ))}

      {selectedQ && (
        <div className="mt-6">
          <h2 className="font-semibold mb-1">{selectedQ}</h2>
          <textarea
            className="w-full h-28 p-2 border rounded"
            placeholder="Type your answer..."
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
          />
          <button
            onClick={evaluate}
            className="mt-2 px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Submit Answer
          </button>

          {result && (
            <div className="mt-4 bg-gray-100 p-4 rounded">
              <p>
                💡 <strong>Confidence Score:</strong> {result.confidence_score}
              </p>
              <p className="mt-2 text-sm text-gray-700">
                ✅ <strong>Model Answer:</strong> {result.expected_answer}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
