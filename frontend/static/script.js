document.addEventListener('DOMContentLoaded', () => {
    let totalComparisons = 0;
    let correctPredictions = 0;
    let truePositives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;

    async function fetchImages(imageId) {
        const response = await fetch(`/get_images?image_id=${imageId}`);
        if (!response.ok) throw new Error('Failed to fetch images');
        return await response.json();
    }

    async function generateMasks(image1, image2) {
        const response = await fetch('/generate_masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image1_id: image1, image2_id: image2 }),
        });
        if (!response.ok) throw new Error('Failed to generate masks');
        return await response.json();
    }

    async function compareImages(imageId) {
        const response = await fetch('/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId }),
        });
        if (!response.ok) throw new Error('Failed to compare images');
        return await response.json();
    }

    async function processImages(imageId) {
        const images = await fetchImages(imageId);
        const { image1, image2 } = images;

        // Display images
        document.getElementById('image1').style.backgroundImage = `url(${image1})`;
        document.getElementById('image2').style.backgroundImage = `url(${image2})`;

        // Generate masks
        const masks = await generateMasks(image1, image2);
        document.getElementById('mask1').style.backgroundImage = `url(${masks.mask1})`;
        document.getElementById('mask2').style.backgroundImage = `url(${masks.mask2})`;

        // Compare images
        const comparisonResult = await compareImages(imageId);
        updateLabels(comparisonResult);

        // Calculate metrics
        totalComparisons++;
        if (comparisonResult.predicted === comparisonResult.actual) {
            correctPredictions++;
            if (comparisonResult.predicted === 1) {
                truePositives++;
            }
        } else {
            if (comparisonResult.predicted === 1) {
                falsePositives++;
            } else {
                falseNegatives++;
            }
        }
        updateMetrics();
    }

    function updateLabels(comparisonResult) {
        const actualLabel = document.getElementById('actualLabel');
        const predictedLabel = document.getElementById('predictedLabel');
        const comparisonText = document.getElementById('comparisonResult');
        const totalCountText = document.getElementById('totalCount');

        actualLabel.textContent = `Actual: ${comparisonResult.actual}`;
        predictedLabel.textContent = `Predicted: ${comparisonResult.predicted}`;
        comparisonText.textContent = `Comparison Result: ${comparisonResult.predicted === comparisonResult.actual ? 'Match' : 'No Match'}`;

        // Show result with a fade-in effect
        comparisonText.style.opacity = 0;
        setTimeout(() => {
            comparisonText.style.opacity = 1;
        }, 100); // Fade in duration
        totalCountText.textContent = `Total Comparisons: ${totalComparisons}`;
    }

    function updateMetrics() {
        const accuracy = (correctPredictions / totalComparisons) * 100;
        const precision = (truePositives / (truePositives + falsePositives)) * 100 || 0; // Avoid division by zero
        const recall = (truePositives / (truePositives + falseNegatives)) * 100 || 0; // Avoid division by zero

        const accuracyText = document.getElementById('accuracyLabel');
        const precisionText = document.getElementById('precisionLabel');
        const recallText = document.getElementById('recallLabel');

        accuracyText.textContent = `Accuracy: ${accuracy.toFixed(2)}%`;
        precisionText.textContent = `Precision: ${precision.toFixed(2)}%`;
        recallText.textContent = `Recall: ${recall.toFixed(2)}%`;
    }

    // Initial call to start processing images
    processImages(1); // Change this to the desired starting image ID
});
