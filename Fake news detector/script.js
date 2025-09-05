// Initialize the text form submission handler
document.getElementById('textForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = document.getElementById('textInput').value;
    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }
    await analyzeText(text);
});

// Initialize the URL form submission handler
document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = document.getElementById('urlInput').value;
    if (!url) {
        alert('Please enter a URL to analyze');
        return;
    }
    await analyzeUrl(url);
});

// Function to analyze text using Google Search
async function analyzeText(text) {
    try {
        showLoading();
        
        // Create search query from text (use first 100 characters)
        const searchQuery = text.substring(0, 100);
        
        // Use Google Custom Search API
        const apiKey = 'AIzaSyDjrV0Xa1fL3PDoKo-TdaJdCwqcV7n6_Lg';
        const searchEngineId = '57004342f8df0435c';
        const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=${searchEngineId}&q=${encodeURIComponent(searchQuery)}`;
        
        const response = await fetch(searchUrl);
        const data = await response.json();
        
        // Analyze search results
        const result = analyzeSearchResults(data, text);
        displayResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        handleError('Error analyzing text. Please try again.');
    } finally {
        hideLoading();
    }
}

// Function to analyze URL
async function analyzeUrl(url) {
    try {
        showLoading();
        
        // First fetch the content from the URL
        const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
        const response = await fetch(proxyUrl);
        
        if (!response.ok) {
            throw new Error('Failed to fetch URL content');
        }
        
        const data = await response.json();
        
        // Extract text content and analyze
        const parser = new DOMParser();
        const doc = parser.parseFromString(data.contents, 'text/html');
        const textContent = doc.body.textContent.trim();
        
        await analyzeText(textContent);
    } catch (error) {
        console.error('Error:', error);
        handleError('Error analyzing URL. Please check the URL and try again.');
    }
}

// Function to analyze search results
function analyzeSearchResults(searchData, originalText) {
    let credibilityScore = 0;
    let matchCount = 0;
    let reputableSourceCount = 0;
    
    // List of reputable news sources
    const reputableSources = [
        'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 'wsj.com',
        'theguardian.com', 'washingtonpost.com', 'bloomberg.com', 'npr.org'
    ];
    
    // Analyze each search result
    if (searchData.items && searchData.items.length > 0) {
        searchData.items.forEach(item => {
            // Check if content appears on reputable sources
            if (reputableSources.some(source => item.link.includes(source))) {
                reputableSourceCount++;
                credibilityScore += 20;
            }
            
            // Check content similarity
            if (calculateSimilarity(item.snippet, originalText) > 0.7) {
                matchCount++;
                credibilityScore += 10;
            }
        });
    }
    
    // Calculate final scores
    const totalScore = Math.min(100, credibilityScore);
    const isFake = totalScore < 50;
    
    // Determine confidence and sentiment
    const confidence = Math.abs(50 - totalScore) * 2;
    let sentiment;
    if (totalScore >= 80) sentiment = 'Highly Credible';
    else if (totalScore >= 60) sentiment = 'Probably Credible';
    else if (totalScore >= 40) sentiment = 'Uncertain';
    else if (totalScore >= 20) sentiment = 'Possibly Fake';
    else sentiment = 'Likely Fake';
    
    return {
        isFake,
        confidence,
        sentiment,
        matchCount,
        reputableSourceCount
    };
}

// Function to calculate text similarity (simple implementation)
function calculateSimilarity(text1, text2) {
    const words1 = text1.toLowerCase().split(/\W+/);
    const words2 = text2.toLowerCase().split(/\W+/);
    const commonWords = words1.filter(word => words2.includes(word));
    return commonWords.length / Math.max(words1.length, words2.length);
}

// Function to display the result
function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const labelElement = document.getElementById('label');
    const confidenceElement = document.getElementById('confidence');
    const sentimentElement = document.getElementById('sentiment');

    // Update the table with results
    const tableBody = document.querySelector('tbody');
    const row = tableBody.querySelector('tr');
    row.cells[1].textContent = result.isFake ? 'FAKE' : 'REAL';
    row.cells[2].textContent = `${result.confidence.toFixed(1)}%`;
    row.cells[3].textContent = result.sentiment;

    // Show the result section
    resultDiv.classList.remove('hidden');
    labelElement.textContent = `This content appears to be ${result.isFake ? 'FAKE' : 'REAL'}`;
    labelElement.textContent += ` (Found ${result.matchCount} similar sources, ${result.reputableSourceCount} from reputable sources)`;
    confidenceElement.textContent = `${result.confidence.toFixed(1)}%`;
    sentimentElement.textContent = result.sentiment;
}

// Error handling function
function handleError(message) {
    const resultDiv = document.getElementById('result');
    const labelElement = document.getElementById('label');
    const confidenceElement = document.getElementById('confidence');
    const sentimentElement = document.getElementById('sentiment');
    
    resultDiv.classList.remove('hidden');
    labelElement.textContent = message;
    confidenceElement.textContent = 'N/A';
    sentimentElement.textContent = 'Error';
    
    // Update table with error state
    const tableBody = document.querySelector('tbody');
    const row = tableBody.querySelector('tr');
    row.cells[1].textContent = 'Error';
    row.cells[2].textContent = 'N/A';
    row.cells[3].textContent = 'Error';
}

// Loading state functions
function showLoading() {
    document.body.style.cursor = 'wait';
    const buttons = document.querySelectorAll('button[type="submit"]');
    buttons.forEach(button => {
        button.disabled = true;
        button.textContent = 'Analyzing...';
    });
}

function hideLoading() {
    document.body.style.cursor = 'default';
    const buttons = document.querySelectorAll('button[type="submit"]');
    buttons.forEach(button => {
        button.disabled = false;
        button.textContent = button.form.id === 'textForm' ? 'Analyze Text' : 'Analyze URL';
    });
}

// Function to fetch and display latest news
async function fetchLatestNews() {
    try {
        const response = await fetch('https://newsapi.org/v2/top-headlines?country=us&apiKey=4f96c50d511a47e48194172e7f1c59f9 ');
        const data = await response.json();
        
        const newsContainer = document.getElementById('news-container');
        newsContainer.innerHTML = '';

        data.articles.slice(0, 6).forEach(article => {
            const newsItem = document.createElement('div');
            newsItem.className = 'p-4 md:w-1/3';
            newsItem.innerHTML = `
                <div class="h-full border-2 border-gray-200 border-opacity-60 rounded-lg overflow-hidden">
                    <img class="lg:h-48 md:h-36 w-full object-cover object-center" src="${article.urlToImage || 'placeholder.jpg'}" alt="news">
                    <div class="p-6">
                        <h2 class="tracking-widest text-xs title-font font-medium text-gray-400 mb-1">${article.source.name}</h2>
                        <h1 class="title-font text-lg font-medium text-gray-900 mb-3">${article.title}</h1>
                        <p class="leading-relaxed mb-3">${article.description || ''}</p>
                        <div class="flex items-center flex-wrap">
                            <a href="${article.url}" target="_blank" class="text-indigo-500 inline-flex items-center md:mb-2 lg:mb-0">Learn More
                                <svg class="w-4 h-4 ml-2" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M5 12h14"></path>
                                    <path d="M12 5l7 7-7 7"></path>
                                </svg>
                            </a>
                        </div>
                    </div>
                </div>
            `;
            newsContainer.appendChild(newsItem);
        });
    } catch (error) {
        console.error('Error fetching news:', error);
    }
}

// Load latest news when the page loads
document.addEventListener('DOMContentLoaded', fetchLatestNews);
  