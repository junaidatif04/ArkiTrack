// Main JavaScript file for frontend interactions
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.querySelector('.upload-box');
    const validateBtn = document.querySelector('.validate-btn');
    const progressBar = document.querySelector('.progress-bar');
    const resultArea = document.getElementById('resultArea');
    let selectedFile = null;

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle drag and drop
    if (uploadBox) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadBox.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => {
                uploadBox.classList.add('highlight');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => {
                uploadBox.classList.remove('highlight');
            });
        });

        uploadBox.addEventListener('drop', handleDrop);
    }

    // Handle file drop
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFileSelection(file);
    }

    // Handle click to upload
    const uploadButton = uploadBox?.querySelector('button');
    if (uploadButton) {
        uploadButton.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => handleFileSelection(e.target.files[0]);
            input.click();
        });
    }

    // Handle file selection
    function handleFileSelection(file) {
        if (!file) return;
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Please select a valid image file (PNG, JPG, or JPEG)');
            return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showError('File size should not exceed 5MB');
            return;
        }

        selectedFile = file;
        uploadBox.innerHTML = `
            <div class="selected-file">
                <i class="fas fa-file-image"></i>
                <span>${file.name}</span>
                <button class="remove-file" onclick="removeFile()">×</button>
            </div>`;
    }

    // Handle validate button click
    if (validateBtn) {
        validateBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            const projectId = document.getElementById('project-id')?.value;
            if (!projectId) {
                showError('Project ID is required');
                return;
            }

            const stage = document.getElementById('stage')?.value;
            if (!stage) {
                showError('Please select a construction stage');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('project_id', projectId);
            formData.append('stage', stage);
            formData.append('proceed', document.getElementById('proceed')?.checked || false);
            formData.append('describe', document.getElementById('describe')?.checked || false);

            try {
                showProgress('Validating image...');
                const response = await fetch('/validate_image', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const result = await response.json();
                hideProgress();

                if (result.success) {
                    showSuccess(result);
                } else if (result.warning) {
                    showWarning(result);
                } else {
                    showError(result.error || 'Validation failed');
                }
            } catch (error) {
                hideProgress();
                showError('Error during validation: ' + error.message);
            }
        });
    }

    // UI Helper Functions
    function showProgress(message) {
        progressBar.style.display = 'block';
        progressBar.innerHTML = `
            <div class="progress">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 100%">
                    ${message}
                </div>
            </div>`;
    }

    function hideProgress() {
        progressBar.style.display = 'none';
    }

    function showSuccess(result) {
        resultArea.innerHTML = `
            <div class="alert alert-success">
                <h5>Validation Successful!</h5>
                <p>${result.message}</p>
                <hr>
                <p><strong>Stage:</strong> ${result.stage}</p>
                <p><strong>Sub-stage:</strong> ${result.sub_stage}</p>
                <p><strong>Confidence:</strong> ${result.confidence}%</p>
                ${result.description ? `<p><strong>AI Description:</strong> ${result.description}</p>` : ''}
                <p class="text-muted">Validation ID: ${result.validation_id}</p>
            </div>`;
    }

    function showWarning(result) {
        resultArea.innerHTML = `
            <div class="alert alert-warning">
                <h5>Validation Warning</h5>
                <p>${result.message}</p>
            </div>`;
    }

    function showError(message) {
        resultArea.innerHTML = `
            <div class="alert alert-danger">
                <h5>Error</h5>
                <p>${message}</p>
            </div>`;
    }

    // Global function to remove selected file
    window.removeFile = function() {
        selectedFile = null;
        uploadBox.innerHTML = `
            <button type="button" class="btn btn-outline-primary">
                <i class="fas fa-cloud-upload-alt"></i> Choose or drag image
            </button>`;
    };

    // Global function to proceed with validation despite warnings
    window.proceedAnyway = function() {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('project_id', document.getElementById('project-id').value);
        formData.append('stage', document.getElementById('stage').value);
        formData.append('describe', 'true');
        formData.append('proceed', 'true');

        validateImage(formData);
    };
});

// Image preview functionality
document.getElementById('image')?.addEventListener('change', function(e) {
    const preview = document.getElementById('imagePreview');
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = 'block';
    }

    if (file) {
        reader.readAsDataURL(file);
    }
});

// Form submission handling
document.getElementById('validationForm')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const resultArea = document.getElementById('resultArea');
    
    try {
        resultArea.innerHTML = '<div class="alert alert-info">Processing image...</div>';
        
        const response = await fetch(form.action, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            if (result.success) {
                let html = '<div class="card">';
                html += '<div class="card-body">';
                html += '<h5 class="card-title">Validation Results</h5>';
                
                // Add stage classification results
                html += '<h6>Stage Classification:</h6>';
                html += `<p>Primary Stage: ${result.primary_stage || result.stage}<br>`;
                html += `Specific Classification: ${result.specific_classification || result.predicted_class}</p>`;
                
                // Add confidence scores if available
                if (result.confidence_scores) {
                    html += '<h6>Confidence Scores:</h6>';
                    html += '<ul>';
                    for (const [key, value] of Object.entries(result.confidence_scores)) {
                        html += `<li>${key}: ${value}%</li>`;
                    }
                    html += '</ul>';
                }
                
                // Add image description if available
                if (result.description) {
                    html += '<h6>Image Description:</h6>';
                    html += `<p>${result.description}</p>`;
                }
                
                // Add message from server
                if (result.message) {
                    html += '<div class="alert alert-success mt-3">';
                    html += `<p class="mb-0">${result.message}</p>`;
                    html += '</div>';
                }
                
                html += '</div></div>';
                resultArea.innerHTML = html;
            } else if (result.message && !result.success) {
                // Handle stage mismatch warning
                resultArea.innerHTML = `
                    <div class="alert alert-warning">
                        <p>${result.message}</p>
                    </div>
                `;
            } else {
                resultArea.innerHTML = `<div class="alert alert-danger">${result.error || 'An error occurred during validation'}</div>`;
            }
        } else {
            // Handle HTTP error responses
            resultArea.innerHTML = `<div class="alert alert-danger">${result.error || 'Server error occurred during validation'}</div>`;
        }
    } catch (error) {
        resultArea.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});

// Progress comparison functionality
const compareBtn = document.getElementById('compareBtn');
if (compareBtn) {
    compareBtn.addEventListener('click', compareProgress);
}

function compareProgress() {
    const projectId = document.getElementById('project-id').value;
    const previousValidationId = document.getElementById('previous-validation').value;
    const currentValidationId = document.getElementById('current-validation').value;

    if (!projectId || !previousValidationId || !currentValidationId) {
        showAlert('Please select both validations to compare', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('previous_doc_id', previousValidationId);
    formData.append('current_doc_id', currentValidationId);

    fetch('/compare', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const comparisonResults = document.getElementById('comparison-results');
            comparisonResults.innerHTML = '';

            // Create container for the entire comparison section
            const container = document.createElement('div');
            container.className = 'comparison-section';
            
            // Create image comparison container
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-comparison-container';
            
            // Previous validation image
            const prevImageDiv = document.createElement('div');
            prevImageDiv.className = 'validation-image previous';
            const prevImage = document.createElement('img');
            prevImage.src = data.previous.image_path ? `/static/${data.previous.image_path}` : '';
            prevImage.alt = 'Previous validation image';
            const prevLabel = document.createElement('p');
            prevLabel.textContent = `Previous: ${data.previous.stage} (${data.previous.sub_stage || 'N/A'})`;
            prevImageDiv.appendChild(prevImage);
            prevImageDiv.appendChild(prevLabel);
            
            // Current validation image
            const currImageDiv = document.createElement('div');
            currImageDiv.className = 'validation-image current';
            const currImage = document.createElement('img');
            currImage.src = data.current.image_path ? `/static/${data.current.image_path}` : '';
            currImage.alt = 'Current validation image';
            const currLabel = document.createElement('p');
            currLabel.textContent = `Current: ${data.current.stage} (${data.current.sub_stage || 'N/A'})`;
            currImageDiv.appendChild(currImage);
            currImageDiv.appendChild(currLabel);
            
            imageContainer.appendChild(prevImageDiv);
            imageContainer.appendChild(currImageDiv);
            container.appendChild(imageContainer);

            // Progress information
            const progressInfo = document.createElement('div');
            progressInfo.className = 'progress-info';
            
            // Progress message
            const messageDiv = document.createElement('div');
            messageDiv.className = 'progress-message';
            messageDiv.innerHTML = data.progress_message.replace(/\n/g, '<br>');
            
            progressInfo.appendChild(messageDiv);
            container.appendChild(progressInfo);
            
            comparisonResults.appendChild(container);
            comparisonResults.style.display = 'block';
        } else {
            showAlert(data.error || 'Error comparing progress', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error comparing progress', 'error');
    });
}

function getStatusIcon(status) {
    switch (status) {
        case 'advanced':
            return 'fas fa-arrow-up text-success';
        case 'regressed':
            return 'fas fa-arrow-down text-danger';
        case 'same':
            return 'fas fa-equals text-warning';
        case 'invalid':
            return 'fas fa-exclamation-triangle text-danger';
        default:
            return 'fas fa-question text-secondary';
    }
}

// UI Helper Functions
function showProgress(message) {
    const progressDiv = document.createElement('div');
    progressDiv.id = 'progressMessage';
    progressDiv.className = 'alert alert-info';
    progressDiv.textContent = message;
    document.body.appendChild(progressDiv);
}

function hideProgress() {
    const progressDiv = document.getElementById('progressMessage');
    if (progressDiv) {
        progressDiv.remove();
    }
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    document.body.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}
