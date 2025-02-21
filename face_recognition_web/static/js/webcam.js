document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('video-feed');
    const videoContainer = document.getElementById('video-container');
    const startButton = document.getElementById('start-camera');
    const stopButton = document.getElementById('stop-camera');
    const overlayElement = document.getElementById('overlay');
    
    let stream = null;
    let isProcessing = false;
    let processingInterval = null;
    
    // Start camera
    startButton.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                } 
            });
            
            videoElement.srcObject = stream;
            startButton.disabled = true;
            stopButton.disabled = false;
            
            // Start processing frames
            processingInterval = setInterval(processFrame, 200); // Process every 200ms
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Cannot access camera. Please check permissions.');
        }
    });
    
    // Stop camera
    stopButton.addEventListener('click', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            clearInterval(processingInterval);
            
            // Clear face boxes
            overlayElement.innerHTML = '';
            
            startButton.disabled = false;
            stopButton.disabled = true;
        }
    });
    
    // Process video frame
    async function processFrame() {
        if (isProcessing || !stream) return;
        
        isProcessing = true;
        
        try {
            // Create canvas to capture frame
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            // Get frame data as base64
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for processing
            const response = await fetch('/process-frame/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame: frameData })
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayFaces(result.faces);
            } else {
                console.error('Error processing frame:', result.error);
            }
            
        } catch (error) {
            console.error('Error processing frame:', error);
        } finally {
            isProcessing = false;
        }
    }
    
    // Display detected faces
    function displayFaces(faces) {
        // Clear previous faces
        overlayElement.innerHTML = '';
        
        const containerRect = videoContainer.getBoundingClientRect();
        const videoRect = videoElement.getBoundingClientRect();
        
        // Scale factors (in case video is resized)
        const scaleX = videoRect.width / videoElement.videoWidth;
        const scaleY = videoRect.height / videoElement.videoHeight;
        
        faces.forEach(face => {
            const [x, y, w, h] = face.box;
            
            // Create box element
            const boxElement = document.createElement('div');
            boxElement.className = 'face-box';
            boxElement.style.left = `${x * scaleX}px`;
            boxElement.style.top = `${y * scaleY}px`;
            boxElement.style.width = `${w * scaleX}px`;
            boxElement.style.height = `${h * scaleY}px`;
            
            // Create name label
            const nameElement = document.createElement('div');
            nameElement.className = 'face-label';
            nameElement.textContent = face.name;
            nameElement.style.left = `${x * scaleX}px`;
            nameElement.style.top = `${(y * scaleY) - 20}px`;
            
            // Create confidence label
            const confidenceElement = document.createElement('div');
            confidenceElement.className = 'confidence-label';
            confidenceElement.textContent = face.confidence;
            confidenceElement.style.left = `${x * scaleX}px`;
            confidenceElement.style.top = `${(y + h) * scaleY + 5}px`;
            
            // Add elements to overlay
            overlayElement.appendChild(boxElement);
            overlayElement.appendChild(nameElement);
            overlayElement.appendChild(confidenceElement);
        });
    }
});