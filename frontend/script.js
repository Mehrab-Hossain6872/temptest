// Initialize the map - centered on Amsterdam (matching the backend bbox)
const map = L.map('map').setView([52.375, 4.860], 13);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Global variables
let startMarker = null;
let endMarker = null;
let routeLayers = [];
let clickCount = 0;
let startCoords = null;
let endCoords = null;

// Mode colors matching the backend
const modeColor = {
    walk: '#FFD700',    // Gold/Yellow
    bike: '#FF8C00',    // Dark Orange
    car: '#4169E1',     // Royal Blue
    transfer: '#808080'  // Gray
};

// Clear all route data and markers
function clearRoute() {
    // Remove all route layers
    routeLayers.forEach(layer => map.removeLayer(layer));
    routeLayers = [];
    
    // Remove markers
    if (startMarker) map.removeLayer(startMarker);
    if (endMarker) map.removeLayer(endMarker);
    
    // Clear info display
    document.getElementById('route-info').innerHTML = '';
    
    // Reset state
    clickCount = 0;
    startCoords = null;
    endCoords = null;
    startMarker = null;
    endMarker = null;
}

// Handle map clicks for selecting start/end points
map.on('click', function(e) {
    if (clickCount === 0) {
        // First click - set start point
        clearRoute();
        startCoords = e.latlng;
        startMarker = L.marker(startCoords, {
            title: 'Start Point'
        }).addTo(map);
        startMarker.bindPopup('Start Point').openPopup();
        clickCount = 1;
        
        // Update info display
        document.getElementById('route-info').innerHTML = 'Click on the map to select destination';
        
    } else if (clickCount === 1) {
        // Second click - set end point and calculate route
        endCoords = e.latlng;
        endMarker = L.marker(endCoords, {
            title: 'End Point'
        }).addTo(map);
        endMarker.bindPopup('End Point').openPopup();
        clickCount = 2;
        
        // Update info display
        document.getElementById('route-info').innerHTML = 'Calculating route...';
        
        // Calculate and display route
        getRoute();
    } else {
        // Third click - reset and start over
        clearRoute();
        startCoords = e.latlng;
        startMarker = L.marker(startCoords, {
            title: 'Start Point'
        }).addTo(map);
        startMarker.bindPopup('Start Point').openPopup();
        clickCount = 1;
        
        document.getElementById('route-info').innerHTML = 'Click on the map to select destination';
    }
});

// Fetch route from backend API
function getRoute() {
    console.log('Fetching route from backend...');
    const url = `http://localhost:8000/route?start_lat=${startCoords.lat}&start_lon=${startCoords.lng}&end_lat=${endCoords.lat}&end_lon=${endCoords.lng}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Route data received:', data);
            drawRoute(data.segments);
            updateRouteInfo(data);
        })
        .catch(error => {
            console.error('Route fetch error:', error);
            document.getElementById('route-info').innerHTML = 
                'Error fetching route. Make sure backend is running on localhost:8000';
        });
}

// Draw route segments on the map
function drawRoute(segments) {
    segments.forEach((segment, index) => {
        // Use coordinates as-is ([lat, lon])
        const latlngs = segment.coords;
        const color = modeColor[segment.mode] || '#000000';
        
        // Create polyline for this segment
        const polyline = L.polyline(latlngs, {
            color: color,
            weight: 6,
            opacity: 0.8,
            smoothFactor: 1
        }).addTo(map);
        
        // Add popup with segment info
        const popupContent = `
            <strong>Mode:</strong> ${segment.mode.charAt(0).toUpperCase() + segment.mode.slice(1)}<br>
            <strong>Time:</strong> ${segment.time.toFixed(1)} min<br>
            <strong>Cost:</strong> ${segment.cost} ৳
        `;
        polyline.bindPopup(popupContent);
        
        routeLayers.push(polyline);
    });
    
    // Fit map to show entire route
    if (routeLayers.length > 0) {
        const group = new L.featureGroup(routeLayers);
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

// Update route information display
function updateRouteInfo(data) {
    const routeInfo = document.getElementById('route-info');
    
    let html = `
        <div style="margin-bottom: 8px;">
            <strong>Total Time:</strong> ${data.total_time} minutes | 
            <strong>Total Cost:</strong> ${data.total_cost} ৳
        </div>
        <div class="mode-legend">
    `;
    
    // Add legend for modes used in this route
    const modesUsed = [...new Set(data.segments.map(s => s.mode))];
    modesUsed.forEach(mode => {
        if (mode !== 'transfer') {
            html += `
                <div class="mode-legend-item">
                    <div class="mode-legend-color" style="background-color: ${modeColor[mode]};"></div>
                    <span>${mode.charAt(0).toUpperCase() + mode.slice(1)}</span>
                </div>
            `;
        }
    });
    
    html += '</div>';
    routeInfo.innerHTML = html;
}

// Add instructions overlay
function addInstructions() {
    const instructionsDiv = document.createElement('div');
    instructionsDiv.id = 'instructions';
    instructionsDiv.innerHTML = `
        <h4>🗺️ How to use:</h4>
        <ul>
            <li>Click once to set start point</li>
            <li>Click again to set destination</li>
            <li>View multimodal route with colors:</li>
            <li style="color: #FFD700;">🚶 Walk - Yellow</li>
            <li style="color: #FF8C00;">🚲 Bike - Orange</li>
            <li style="color: #4169E1;">🚗 Car - Blue</li>
            <li>Click anywhere to start over</li>
        </ul>
    `;
    document.body.appendChild(instructionsDiv);
}

// Initialize instructions when page loads
document.addEventListener('DOMContentLoaded', function() {
    addInstructions();
    document.getElementById('route-info').innerHTML = 'Click on the map to select start point';
});

// Add keyboard shortcut to clear route
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        clearRoute();
        document.getElementById('route-info').innerHTML = 'Click on the map to select start point';
    }
});

// Add double-click to clear route
map.on('dblclick', function(e) {
    clearRoute();
    document.getElementById('route-info').innerHTML = 'Click on the map to select start point';
});