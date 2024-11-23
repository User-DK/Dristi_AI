import React, { useState } from 'react';
import './App.css'

function App() {
    const [audioSrc, setAudioSrc] = useState(null);
    const [imageDescription, setImageDescription] = useState('');
    const [translatedText, setTranslatedText] = useState('');

    // Function to handle image upload
    const handleImageUpload = async (e) => {
        const formData = new FormData();
        formData.append('image', e.target.files[0]); // Append the image to the form data

        try {
            // Send the image to the backend
            const response = await fetch('http://localhost:3000/generate-description', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json(); // Parse the JSON response
                setImageDescription(data.description); // Set the image description
                setTranslatedText(data.translated_text); // Set the translated text

                // Create an audio URL from the Base64 data
                const audioUrl = `data:audio/wav;base64,${data.audio_data}`;
                setAudioSrc(audioUrl); // Set the audio URL for playback
            } else {
                console.error('Error generating audio description');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };

    return (
        <div>
            <h1>Dristi AI</h1>
            {/* Input for image file upload */}
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            
            {/* Display the image description and translated text */}
            {imageDescription && (
                <div>
                    <h2>Description:</h2>
                    <p>{imageDescription}</p>
                </div>
            )}
            {translatedText && (
                <div>
                    <h2>Translated Text:</h2>
                    <p>{translatedText}</p>
                </div>
            )}

            {/* Audio element to play the generated description */}
            {audioSrc && <audio controls src={audioSrc} />}
        </div>
    );
}

export default App;
