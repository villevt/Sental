import React, { useEffect } from 'react';
import { useState } from 'react';
import './App.css';
import axios from "axios";

function App() {
  const [sentiment, setSentiment] = useState("❔")
  const [probability, setProbability] = useState("❔");
  const [timer, setTimer] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingPeriods, setLoadingPeriods] = useState(0);
  const [textLength, setTextLength] = useState(0);

  useEffect(() => {
    window.setTimeout(() => {
      if (loading) {
        setLoadingPeriods(loadingPeriods + 1);
      }
    }, 500);
  }, [loading, loadingPeriods]);

  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>): void => {
    if (timer) {
      clearTimeout(timer)
    }

    // Delay API query to make sure the user stopped typing
    const text = e.currentTarget.value;
    const t = window.setTimeout(() => {
      e.preventDefault();

      if (text.length > 0 && text.length < 401) {
        axios.post(process.env.REACT_APP_BACKEND_URL + "/classify/", { text })
          .then(res => {
            setLoading(false);
            const positive = res.data.positive;
            if (positive) {
              setSentiment("🙂");
            } else {
              setSentiment("☹️");
            }

            const probability = (res.data.probability * 100).toFixed(2) + "%";
            setProbability(probability)
          })
          .catch(error => {
            setLoading(false);
            alert(error.message)
          })
      } else {
        setSentiment("❔");
        setProbability("❔");
      }
    }, 2000);

    setTimer(t);

    setTextLength(text.length);

    // Simple visual effect for loading    
    if (text.length === 0 || text.length > 400) {
      setLoading(false);
    } else if (!loading) {
      setLoading(true);
    }
  }

  return (
    <div className="App">
      <header>
        <img src={require("./logo-full.png")} className="header-logo" alt="logo" />
      </header>
      <main>
        <div id="text-input-col">
          <b>How does your text sound?</b>
          <textarea placeholder="Start typing..." onChange={handleInput} />
          <p className={textLength > 400 ? "text-warning" : "text-normal"}>{textLength} / 400 characters</p>
        </div>
        <div id="results">
          <table>
            <tbody>
              <tr id="results-sentiment">
                <td>Sentiment</td>
                <td>
                  {loading ? ".".repeat(loadingPeriods % 3 + 1) : sentiment}
                </td>
              </tr>
              <tr id="results-probability">
                <td>Probability</td>
                <td>
                  {loading ? ".".repeat(loadingPeriods % 3 + 1) : probability}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </main>
      <footer>
        <a href="https://github.com/villevt/Sentimentai">
          Source in <img src={require("./GitHub-Mark-Light-120px-plus.png")} alt="github logo"/>
        </a>
      </footer>
    </div>
  );
}

export default App;
