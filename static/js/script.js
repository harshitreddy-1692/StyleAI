
document.addEventListener("DOMContentLoaded", function () {

    const imageInput = document.getElementById("imageInput");
    const browseBtn = document.getElementById("browseBtn");
    const uploadBox = document.getElementById("uploadBox");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeBtn = document.getElementById("removeBtn");
    const searchBtn = document.getElementById("searchBtn");
    const statusMessage = document.getElementById("statusMessage");

    if (!imageInput || !browseBtn || !uploadBox ||
        !previewContainer || !imagePreview ||
        !removeBtn || !searchBtn || !statusMessage) {
        console.error("Required HTML elements are missing.");
        return;
    }

    let selectedFile = null;

    browseBtn.addEventListener("click", function () {
        imageInput.click();
    });

    imageInput.addEventListener("change", function () {
        const file = imageInput.files[0];
        if (file) showPreview(file);
    });

    function showPreview(file) {
        if (!file.type.startsWith("image/")) {
            statusMessage.textContent = "Please select a valid image.";
            return;
        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = function (event) {
            imagePreview.src = event.target.result;
            previewContainer.style.display = "block";
            searchBtn.disabled = false;
            statusMessage.textContent = "Image selected successfully!";
        };

        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", function () {
        selectedFile = null;
        imageInput.value = "";
        imagePreview.src = "";
        previewContainer.style.display = "none";
        searchBtn.disabled = true;
        statusMessage.textContent = "Choose another image.";
    });

    uploadBox.addEventListener("dragover", function (event) {
        event.preventDefault();
        uploadBox.classList.add("drag-over");
    });

    uploadBox.addEventListener("dragleave", function () {
        uploadBox.classList.remove("drag-over");
    });

    uploadBox.addEventListener("drop", function (event) {
        event.preventDefault();
        uploadBox.classList.remove("drag-over");

        const file = event.dataTransfer.files[0];
        if (file) showPreview(file);
    });

    searchBtn.addEventListener("click", async function () {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("image", selectedFile);

        searchBtn.disabled = true;
        statusMessage.textContent = "AI is identifying your image...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed.");
            }

            statusMessage.textContent =
                "Identified: " + result.category +
                " | Confidence: " + result.confidence + "%";

        } catch (error) {
            statusMessage.textContent = error.message;
            console.error(error);
        } finally {
            searchBtn.disabled = false;
        }
    });

});
