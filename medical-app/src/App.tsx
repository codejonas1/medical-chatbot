import { useEffect, useState, useRef } from 'react';
import './App.css';

interface IMessage{
  type: "bot" | "user";
  text: string;
}

function App() {
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);

  const scrollableRef = useRef<HTMLDivElement>({} as HTMLDivElement);

  useEffect(() => {
    const scrollableBox = scrollableRef.current;
    scrollableBox.scrollTop = scrollableBox.scrollHeight;
  }, [messages])

  const getData = async () => {
    setLoading(true)
    const url = "http://localhost:5000/chat"

    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");
    myHeaders.append("accept", "text/plain");

    try {
      const response = await fetch(url,
        {
          method: "POST",
          body: JSON.stringify({ query: userInput }),
          headers: myHeaders,
        }
      );
      if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
      }

      const answer = await response.text();
      setMessages(prevMessages => [...prevMessages, {type: "bot", text: answer}]);

    } catch (error) {
      console.error(error);
    } finally{
      setLoading(false)
    }
  }

  const handleMessage = () => {
    if (userInput) {
      setMessages(prevMessages => [...prevMessages, {type: "user", text: userInput}]);
      getData();
    }
  }

  return (
    <div className="App">
      <div className='container'>
        <header className='chat-bar'>
          <div className='chat-bar-logo'></div>
          <h1>First Aid Chatbot</h1>
        </header>
        <div className='messages-box' ref={scrollableRef}>
          {messages.map((item, id) => {
            return (
              <div className={`message-item message-item-${item.type}`}>
                <span className='message-placeholder'></span>
                <span className={`message message-${item.type}`} key={id} >
                  <p>{item.text}</p>
                </span>
              </div>
            )
          })}
          {loading ? <div className={`message-item message-item-bot`}>
                <span className='message-placeholder'></span>
                <span className={`message message-bot`} >
                  <p>writing</p>
                </span>
              </div> : ""}
        </div>
        <div className='input-form'>
          <form onSubmit={e => {
            e.preventDefault();
            handleMessage();
            setUserInput("");
          }}>
            <input type="text" value={userInput} placeholder="How do you feel?" onChange={(e) => setUserInput(e.target.value)}/>
            <button type="submit">Sent</button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
