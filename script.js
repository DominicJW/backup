async function getAttentionPlots(dataIndex){
    //TODO FIX this for new index system. to be fair does not need updating
    //TODO clear old attention?from the other branches
    //TODO should set data attribute of img
    //we know its token index, but we should also set query token index so we can keep track of what the image means if we copy and pin it.

    try {
        const response = await fetch('/generate_plots_and_return', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({token_index: dataIndex})
        });
        
        if (!response.ok) throw new Error('Network response was not ok');
        
        // Response is a ZIP file, so we use ArrayBuffer
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        
        // Create a new JSZip instance
        const zip = await JSZip.loadAsync(arrayBuffer);
        
        // Process each file in the ZIP
        const imgPromises = [];
        zip.forEach((relativePath, zipEntry) => {

            if (zipEntry.name.endsWith('.png')) { // Only process PNG files
                imgPromises.push(zipEntry.async('base64').then((data) => {
                    console.log(zipEntry.name)
                    var i = zipEntry.name.slice(6,-4);
                    const imgElement = document.querySelector(`.token-container[data-index="${i}"] .token-heatmap`);
                    if (imgElement) {
                        imgElement.src = `data:image/png;base64,${data}`; // Set the img src to the base64 data
                    }
                }));
            }
        });
        // Wait for all images to be processed
        await Promise.all(imgPromises);
    } catch (err) {
        console.error("predict error:", err);
    }
}

let selectedElement = null; // Global variable to hold the reference to the clicked element
function showContextMenu(event) {
    event.stopPropagation();
    event.preventDefault(); // Prevent the default context menu
    selectedElement = event.currentTarget; // Store reference to the clicked element
    const contextMenu = document.getElementById('contextMenu');
    contextMenu.style.display = 'block';
    contextMenu.style.left = `${event.pageX}px`;
    contextMenu.style.top = `${event.pageY}px`;
}


function handleClickAttenion(action) {
    if (selectedElement) {
        const dataIndex = selectedElement.dataset.index;
        getAttentionPlots(dataIndex);
        // document.getElementById('token-index').value = dataIndex;
    }

    hideContextMenu();
}

function handleClick(action) {
    if (selectedElement) {
        const dataIndex = Number(selectedElement.dataset.index);
        console.log(`${action}`)
        // getAttentionPlots(dataIndex);
    }
    hideContextMenu();
}


async function createBranch(argument) {
if (selectedElement) {
    const dataIndex = selectedElement.dataset.index;
    console.log(dataIndex);
    console.log("hello")
    const text = document.getElementById("input_text").value;
    const response = await fetch('/create_new_branch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token_index: dataIndex, input_text: text})
    });
    
    if (!response.ok) {
        console.error('Request failed:', response.status);
        return;
    }
    
    const data = await response.text();
    selectedElement.insertAdjacentHTML('beforeend', data);
}    //get 

    //should return the tokens and the client could organize the html stuff
    hideContextMenu();   
}


function hideContextMenu() {
    const contextMenu = document.getElementById('contextMenu');
    contextMenu.style.display = 'none';
}

// Hide the context menu when clicking elsewhere
window.onclick = function(event) {
    const contextMenu = document.getElementById('contextMenu');
    if (event.target !== contextMenu && !contextMenu.contains(event.target)) {
        hideContextMenu();
    }
};



function setTokenWidths(root = document.getElementById('token-display')) {
  if (!root) return;
  root.querySelectorAll('.token-container').forEach(container => {
    const token = container.querySelector(':scope > .token');
    if (!token) return;
    const rect = token.getBoundingClientRect();
    const width = Math.ceil(rect.width);
    container.style.setProperty('--token-width', width + 'px');
    container.querySelectorAll(':scope > .token-container').forEach(childContainer => {
        childContainer.style.setProperty("--parent-token-width", width + 'px');
    });

  });
}



setTokenWidths();
let _t;
window.addEventListener('resize', () => {
  clearTimeout(_t);
  _t = setTimeout(setTokenWidths, 120);
});