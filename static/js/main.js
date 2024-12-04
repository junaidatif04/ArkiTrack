// Main JavaScript file for frontend interactions
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.querySelector('.upload-box');
    const validateBtn = document.querySelector('.validate-btn');
    let selectedFile = null;

    // Handle click to upload
    const uploadButton = uploadBox.querySelector('button');
    if (uploadButton) {
        uploadButton.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    uploadBox.innerHTML = `<span>Selected: ${file.name}</span>`;
                }
            };
            input.click();
        });
    }

    // Handle validate button click
    if (validateBtn) {
        validateBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('Please select an image first');
                return;
            }

            // For now, just show some dummy text in the textareas
            document.getElementById('validation-result').value = 'Validation in progress...';
            document.getElementById('generated-description').value = 'Generating description...';
        });
    }
});
