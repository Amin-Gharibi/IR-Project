document.getElementById("search-form").addEventListener("submit", async function (e) {
	e.preventDefault();
	const query = document.getElementById("query-input").value;
	const resultsContainer = document.getElementById("results");
	resultsContainer.innerHTML = "Loading...";

	try {
		const res = await fetch("http://localhost:8000/api/search", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ query: query, top_k: 10 }),
		});

		const data = await res.json();

		if (data.status === "success" || data.status === "warning") {
			if (!data.results || data.results.length === 0) {
				resultsContainer.innerHTML = "<p>No results found.</p>";
				return;
			}

			resultsContainer.innerHTML = "";
			data.results.forEach((doc) => {
				const item = document.createElement("div");
				item.className = "result-item";

				const header = document.createElement("h3");
				header.textContent = `Document ID: ${doc.document_id} - Similarity Score: ${doc.similarity_score.toFixed(2)}`;

				const content = document.createElement("p");
				content.style.marginTop = "16px";
				content.textContent = doc.content;

				item.appendChild(header);
				item.appendChild(content);
				resultsContainer.appendChild(item);
			});
		} else {
			resultsContainer.innerHTML = `<p style="color: red;">${data.message}</p>`;
		}
	} catch (error) {
		resultsContainer.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
	}
});
