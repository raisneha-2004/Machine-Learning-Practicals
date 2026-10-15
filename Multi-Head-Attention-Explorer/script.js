
const sentenceInput = document.getElementById("sentence");
const headCountInput = document.getElementById("headCount");
const maskToggle = document.getElementById("maskToggle");
const showNumbersInput = document.getElementById("showNumbers");
const generateBtn = document.getElementById("generateBtn");
const randomBtn = document.getElementById("randomBtn");
const heatmaps = document.getElementById("heatmaps");
const inspector = document.getElementById("inspector");
const tokenCount = document.getElementById("tokenCount");
const errorMessage = document.getElementById("errorMessage");

const HEAD_COLORS = [
    [104, 142, 255],
    [34, 197, 155],
    [190, 112, 255],
    [255, 157, 83],
    [255, 105, 143],
    [68, 200, 220],
    [218, 190, 83],
    [142, 170, 135]
];

let tokens = [];
let heads = [];
let selectedRow = 0;

// Create a random vector.
function randomVector(size) {
    return Array.from(
        { length: size },
        () => Math.random() * 2 - 1
    );
}

// Dot product of two vectors.
function dot(a, b) {
    return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

// Convert scores to probabilities.
function softmax(scores) {
    const max = Math.max(...scores);
    const exps = scores.map(value => Math.exp(value - max));
    const total = exps.reduce((sum, value) => sum + value, 0);

    return exps.map(value => value / total);
}

// Calculate attention separately for each head.
function createHead(tokenList, headIndex, useMask) {
    const dimension = 8;
    const n = tokenList.length;

    // Independent random projection weights per head.
    const WQ = Array.from(
        { length: dimension },
        () => randomVector(dimension)
    );

    const WK = Array.from(
        { length: dimension },
        () => randomVector(dimension)
    );

    const WV = Array.from(
        { length: dimension },
        () => randomVector(dimension)
    );

    // Each token starts with a random embedding.
    const embeddings = tokenList.map(() => randomVector(dimension));

    // Q, K and V are calculated from embeddings and weights.
    function project(vector, weights) {
        return weights.map(row => dot(vector, row) / Math.sqrt(dimension));
    }

    const queries = embeddings.map(vector => project(vector, WQ));
    const keys = embeddings.map(vector => project(vector, WK));
    const values = embeddings.map(vector => project(vector, WV));

    const matrix = [];
    const outputVectors = [];

    for (let i = 0; i < n; i++) {
        let scores = keys.map(key =>
            dot(queries[i], key) / Math.sqrt(dimension)
        );

        // A causal mask prevents looking at future tokens.
        if (useMask) {
            scores = scores.map((score, j) =>
                j > i ? -Infinity : score
            );
        }

        const weights = softmax(scores);
        matrix.push(weights);

        // Weighted sum of value vectors.
        const output = Array(dimension).fill(0);

        for (let j = 0; j < n; j++) {
            for (let d = 0; d < dimension; d++) {
                output[d] += weights[j] * values[j][d];
            }
        }

        outputVectors.push(output);
    }

    return {
        id: headIndex,
        matrix,
        outputVectors
    };
}

function colorForWeight(weight, headIndex) {
    const [r, g, b] = HEAD_COLORS[headIndex % HEAD_COLORS.length];
    const strength = Math.min(1, weight * 1.8);

    const red = Math.round(15 + (r - 15) * strength);
    const green = Math.round(24 + (g - 24) * strength);
    const blue = Math.round(43 + (b - 43) * strength);

    return `rgb(${red}, ${green}, ${blue})`;
}

function renderHeatmap(head, headIndex) {
    const n = tokens.length;
    const card = document.createElement("article");
    card.className = "head-card";

    const title = document.createElement("div");
    title.className = "head-title";

    const heading = document.createElement("h3");
    heading.textContent = `Head ${headIndex + 1}`;

    const label = document.createElement("span");
    label.textContent = "ATTENTION MATRIX";

    title.append(heading, label);

    const scroll = document.createElement("div");
    scroll.className = "heatmap-scroll";

    const grid = document.createElement("div");
    grid.className = "heatmap";
    grid.style.gridTemplateColumns =
        `58px repeat(${n}, 38px)`;

    const corner = document.createElement("div");
    corner.className = "corner";
    corner.textContent = "Query ↓";
    grid.appendChild(corner);

    // Column labels are the tokens being attended to.
    tokens.forEach(token => {
        const labelCell = document.createElement("div");
        labelCell.className = "column-label";
        labelCell.textContent = token;
        labelCell.title = `Key: ${token}`;
        grid.appendChild(labelCell);
    });

    head.matrix.forEach((row, i) => {
        const rowLabel = document.createElement("button");
        rowLabel.type = "button";
        rowLabel.className = "row-label";
        rowLabel.textContent = tokens[i];
        rowLabel.title = `Inspect attention for ${tokens[i]}`;
        rowLabel.setAttribute(
            "aria-label",
            `Inspect token ${tokens[i]}`
        );

        if (i === selectedRow) {
            rowLabel.classList.add("selected");
        }

        rowLabel.addEventListener("click", () => {
            selectedRow = i;
            renderAll();
        });

        grid.appendChild(rowLabel);

        row.forEach((weight, j) => {
            const cell = document.createElement("button");
            cell.type = "button";
            cell.className = "cell";
            cell.style.background = colorForWeight(weight, headIndex);

            if (i === selectedRow) {
                cell.classList.add("selected-row");
            }

            const isMasked = maskToggle.value === "on" && j > i;

            if (isMasked) {
                cell.classList.add("masked");
                cell.disabled = true;
            }

            const percent = (weight * 100).toFixed(1);

            cell.title = isMasked
                ? `${tokens[i]} cannot attend to future token ${tokens[j]}`
                : `${tokens[i]} → ${tokens[j]}: ${percent}% attention`;

            cell.setAttribute(
                "aria-label",
                `${tokens[i]} to ${tokens[j]}, ${percent}% attention`
            );

            if (showNumbersInput.checked) {
                cell.textContent = weight.toFixed(2);
            }

            grid.appendChild(cell);
        });
    });

    scroll.appendChild(grid);
    card.append(title, scroll);
    heatmaps.appendChild(card);
}

function renderInspector() {
    if (!tokens.length || !heads.length) {
        inspector.textContent = "Enter a sentence to begin.";
        return;
    }

    inspector.replaceChildren();

    const intro = document.createElement("p");
    intro.className = "hint";
    intro.textContent =
        `Selected token: "${tokens[selectedRow]}". Strongest attention target for each head:`;
    inspector.appendChild(intro);

    heads.forEach((head, index) => {
        const row = head.matrix[selectedRow];

        let bestIndex = 0;

        for (let j = 1; j < row.length; j++) {
            if (row[j] > row[bestIndex]) {
                bestIndex = j;
            }
        }

        const detail = document.createElement("div");
        detail.className = "inspector-row";

        const strong = document.createElement("strong");
        strong.textContent = `Head ${index + 1}: `;

        const description = document.createElement("span");
        description.textContent =
            `"${tokens[selectedRow]}" looks most strongly at ` +
            `"${tokens[bestIndex]}" (${(row[bestIndex] * 100).toFixed(1)}%).`;

        detail.append(strong, description);
        inspector.appendChild(detail);
    });
}

function renderAll() {
    heatmaps.replaceChildren();

    heads.forEach((head, index) => {
        renderHeatmap(head, index);
    });

    renderInspector();
}

function generateAttention(newWeights = true) {
    errorMessage.textContent = "";

    const sentence = sentenceInput.value.trim();

    if (!sentence) {
        errorMessage.textContent = "Please enter a sentence.";
        return;
    }

    const newTokens = sentence.split(/\s+/);

    if (newTokens.length > 12) {
        errorMessage.textContent =
            "Please use 12 tokens or fewer for this demo.";
        return;
    }

    tokens = newTokens;
    selectedRow = Math.min(selectedRow, tokens.length - 1);

    const count = Number(headCountInput.value);
    const useMask = maskToggle.value === "on";

    // A new generation creates fresh random embeddings and weights.
    // The argument is retained for clarity and future extensions.
    if (newWeights) {
        heads = Array.from(
            { length: count },
            (_, i) => createHead(tokens, i, useMask)
        );
    } else {
        heads = Array.from(
            { length: count },
            (_, i) => createHead(tokens, i, useMask)
        );
    }

    tokenCount.textContent = `${tokens.length} tokens`;
    renderAll();

    // Verify each attention row sums to approximately 1.
    const valid = heads.every(head =>
        head.matrix.every(row => {
            const sum = row.reduce((total, value) => total + value, 0);
            return Math.abs(sum - 1) < 1e-8;
        })
    );

    if (!valid) {
        errorMessage.textContent =
            "Attention probability check failed.";
    }
}

generateBtn.addEventListener("click", () => generateAttention(true));
randomBtn.addEventListener("click", () => generateAttention(true));

headCountInput.addEventListener("change", () => generateAttention(true));
maskToggle.addEventListener("change", () => generateAttention(true));

showNumbersInput.addEventListener("change", () => {
    renderAll();
});

sentenceInput.addEventListener("keydown", event => {
    if (event.ctrlKey && event.key === "Enter") {
        generateAttention(true);
    }
});

// Initial demo.
generateAttention(true);
