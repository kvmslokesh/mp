const socket = io.connect("http://127.0.0.1:5000", {
    transports: ["websocket"]
});

// Display real-time alerts
socket.on("new_attack", function (data) {
    const alertDiv = document.getElementById("alerts");
    const alert = document.createElement("div");
    alert.className = "alert";
    alert.innerHTML = `Attack detected with confidence: ${data.confidence}`;
    alertDiv.appendChild(alert);
});

// Fetch and display historical attack data on page load
window.onload = function() {
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            const historyDiv = document.getElementById("history");
            data.forEach(attack => {
                const attackEntry = document.createElement("div");
                attackEntry.className = "alert";
                attackEntry.innerHTML = `Past Attack: Confidence ${attack.prediction.confidence}`;
                historyDiv.appendChild(attackEntry);
            });
        });
};
