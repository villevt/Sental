import React from 'react';
import { useState } from 'react';
import './App.css';
import axios from "axios";

function App() {
  const [sentiment, setSentiment] = useState("Waiting..")
  const [timer, setTimer] = useState(0);

  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>): void => {
    if (timer) {
      clearTimeout(timer)
    }

    // Delay API query to make sure the user stopped typing
    const text = e.currentTarget.value;
    const t = window.setTimeout(() => {
      e.preventDefault();

      if (text.length > 0) {
        axios.post("http://localhost:8000/classify/", { text })
          .then(res => {
            const positive = res.data.positive;
            if (positive) {
              setSentiment("Positive");
            } else {
              setSentiment("Negative");
            }
          })
      }
    }, 2000);

    setSentiment("Wait a moment...")

    setTimer(t);
  }

  return (
    <div className="App">
      <header>
        <img src={require("./logo-full.png")} className="header-logo" alt="logo" />
      </header>
      <main>
        <div id="text-input-col">
          <textarea placeholder="Start typing..." onChange={handleInput} />
        </div>
        <div>
          <b>Sentiment</b>
          <br />
          {sentiment}
        </div>
      </main>
    </div>
  );
}

export default App;
