let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const recordButton = document.getElementById("recordButton");
const audioPlayback = document.getElementById("audioPlayback");

function setButtonIcon(isRecording) {
    const currentIcon = document.querySelector(".icon");
    if (currentIcon) currentIcon.remove();

    const icon = document.createElement("div");
    icon.classList.add("icon");
    icon.classList.add(isRecording ? "stop" : "play");
    recordButton.appendChild(icon);
}

// Initialize with play icon
setButtonIcon(false);

recordButton.addEventListener("mousedown", async () => {
    if (!isRecording) {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        isRecording = true;

        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob);

            const response = await fetch('/process-audio', {
                method: 'POST',
                body: formData
            });

            const processedAudioBlob = await response.blob();
            const audioURL = URL.createObjectURL(processedAudioBlob);

            audioPlayback.src = audioURL;
            audioPlayback.style.display = 'block';
        };

        mediaRecorder.start();
        setButtonIcon(true);
    }
});

recordButton.addEventListener("mouseup", () => {
    if (isRecording) {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        setButtonIcon(false);
    }
});
