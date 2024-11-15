
document.addEventListener('DOMContentLoaded', function() {


     let totalComparisons = 0;
     let correctPredictions = 0;
     let truePositives = 0;
     let falsePositives = 0;
    let  falseNegatives = 0;
   let  currentImageId = 0;  // Starting image ID
    const maxImageId = 100;   // Total number of images to process

    // Fetch images from the server
    async function fetchImages(imageId) {
        const response = await fetch(`/get_images?image_id=${imageId}`);
        if (!response.ok) throw new Error('Failed to fetch images');
        return await response.json();
    }

    // Generate segmentation masks for the images
    async function generateMasks(image1, image2) {
        const response = await fetch('/generate_masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image1_id: image1, image2_id: image2 }),
        });
        if (!response.ok) throw new Error('Failed to generate masks');
        return await response.json();
    }

    // Compare the two images
async function compareImages(imageId) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 13000); // 5 seconds timeout

    try {
        const response = await fetch('/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId }),
            signal: controller.signal, // Link the abort signal
        });

        if (!response.ok) throw new Error('Failed to compare images');

        const comparisonResult = await response.json();
        clearTimeout(timeoutId); // Clear the timeout if successful
        return comparisonResult;
    } catch (error) {
        console.error('Comparison request failed or was aborted:', error);
        // Optionally handle this by showing an error message to the user
        return { actual: null, predicted: null }; // Return null or some default value
    }
}

    // Function to apply a mask to an image

    // Function to handle the processing of a single image pair
    async function processImages(imageId) {
        try {
            // Clear the previous result and mask before loading new images
            resetUI();

            // Fetch the images
            const images = await fetchImages(imageId);
            const { image1, image2 } = images;

            // Display the images in the UI
            document.getElementById('image1').style.backgroundImage = `url(${image1})`;
            document.getElementById('image2').style.backgroundImage = `url(${image2})`;

            // Generate masks
            const masks = await generateMasks(image1, image2);

            // Apply the mask to the images and update the UI
          
            document.getElementById('mask1').style.backgroundImage = `url(${masks.mask1})`;
            document.getElementById('mask2').style.backgroundImage = `url(${masks.mask2})`;

            // Once the masks are applied, perform the image comparison
            const comparisonResult = await compareImages(imageId);

            // Update the UI with the comparison result
      //      updateLabels(comparisonResult);

            // Update accuracy, precision, recall
           await calculateMetrics(comparisonResult);


            // Update the progress bar
            updateProgressBar(totalComparisons + 1, maxImageId); // Update with new total comparisons

await delay(4000);
 
if (imageId < maxImageId) {
            currentImageId++;
            processImages(currentImageId);
        }

        } catch (error) {
            console.error(`Error processing image ${imageId}:`, error);
        }
    }

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


    // Function to reset the UI before loading new images
    function resetUI() {
        document.getElementById('image1').style.backgroundImage = '';
        document.getElementById('image2').style.backgroundImage = '';
        document.getElementById('mask1').style.backgroundImage = '';
        document.getElementById('mask2').style.backgroundImage = '';
        document.getElementById('actualLabel').textContent = 'Actual: ';
        document.getElementById('predictedLabel').textContent = 'Predicted: ';
        document.getElementById('comparisonResult').textContent = 'Comparison Result: ';
    }

/*    // Function to update labels with the comparison result
    function updateLabels(comparisonResult) {
        const actualLabel = document.getElementById('actualLabel');
        const predictedLabel = document.getElementById('predictedLabel');
        const comparisonText = document.getElementById('comparisonResult');
        const totalCountText = document.getElementById('totalCount');

	console.log(comparisonResult)
        actualLabel.textContent = `Actual: ${comparisonResult.actual}`;
        predictedLabel.textContent = `Predicted: ${comparisonResult.predicted}`;
        comparisonText.textContent = `Comparison Result: ${comparisonResult.predicted === comparisonResult.actual ? 'Match' : 'No Match'}`;

        totalComparisons++; // Increment the total comparisons count
        totalCountText.textContent = `Total Comparisons: ${totalComparisons}`;
    }*/

    // Function to calculate and update metrics
    function calculateMetrics(comparisonResult) {
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

       

        const actualLabel = document.getElementById('actualLabel');
        const predictedLabel = document.getElementById('predictedLabel');
        const comparisonText = document.getElementById('comparisonResult');
        const totalCountText = document.getElementById('totalCount');

	console.log(comparisonResult)
        actualLabel.textContent = `Actual: ${comparisonResult.actual}`;
        predictedLabel.textContent = `Predicted: ${comparisonResult.predicted}`;
        comparisonText.textContent = `Comparison Result: ${comparisonResult.predicted == '0' &&  comparisonResult.actual == '0' ? 'Match' : 'No Match'}`;

        totalComparisons++; // Increment the total comparisons count
        totalCountText.textContent = `Total Comparisons: ${totalComparisons}`;


        const accuracy = (correctPredictions / totalComparisons) * 100 || 0; // Avoid division by zero
        const precision = (truePositives / (truePositives + falsePositives)) * 100 || 0; // Avoid division by zero
        const recall = (truePositives / (truePositives + falseNegatives)) * 100 || 0; // Avoid division by zero

        document.getElementById('accuracyLabel').textContent = `Accuracy: ${accuracy.toFixed(2)}%`;
        document.getElementById('precisionLabel').textContent = `Precision: ${precision.toFixed(2)}%`;
        document.getElementById('recallLabel').textContent = `Recall: ${recall.toFixed(2)}%`;

       


    }

    // Function to update the progress bar
    function updateProgressBar(current, total) {
        const progressBar = document.getElementById('progress');
        const percentage = (current / total) * 100;
        progressBar.style.width = `${percentage}%`;
    }



window.startx = function () {


processImages(currentImageId);


}



})
