
# Optimum Price Analysis for Sellers

## Overview

The **Optimum Price Analysis for Sellers** project is a web application designed to help regional vendors set optimal prices for their products. The application leverages machine learning algorithms to predict the best pricing strategies based on data gathered from Amazon, such as product prices, ratings, and reviews. The goal is to enable sellers to make informed decisions that enhance their competitive edge in the online marketplace.

## Features

- **Data Collection**: Gathers data on product prices, ratings, and review counts from Amazon.
- **Data Processing**: Cleans and normalizes the data to ensure accuracy for analysis.
- **Feature Extraction**: Extracts important features like star ratings and the number of reviews.
- **Machine Learning**: Utilizes Ridge Regression to predict optimal prices based on extracted features.
- **Real-time Prediction**: Offers real-time recommendations for optimal pricing strategies.
- **Interactive UI**: Built with Angular, allowing sellers to input product details and receive data-driven pricing suggestions.
- **Visual Gallery**: Displays images of best-selling products, providing practical examples of effective presentation and positioning.

## Technologies Used

- **Frontend**: Angular
- **Backend**: Python
- **Machine Learning**: Scikit-learn (Ridge Regression)
- **Data Scraping**: Python (BeautifulSoup, Requests)
- **Deployment**: Docker (optional)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/optimum-price-analysis.git
    cd optimum-price-analysis
    ```

2. **Install backend dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install frontend dependencies**:
    ```bash
    cd frontend
    npm install
    ```

4. **Run the application**:
    - **Backend**:
        ```bash
        python app.py
        ```
    - **Frontend**:
        ```bash
        ng serve
        ```

    The application will be available at `http://localhost:4200`.

## Usage

- **Input product details**: Enter the product's existing price, number of reviews, and ratings.
- **Receive recommendations**: The system will provide optimal pricing suggestions in real-time.
- **Explore best practices**: View the visual gallery for examples of best-selling products.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit the changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, feel free to open an issue on GitHub or contact the project maintainers:

- **Rohan Sethi** - [LinkedIn](https://www.linkedin.com/in/rohan-sethi-a27107178/) | [GitHub](https://github.com/rohansethi1999430)

---

You can customize the above template based on your project's specific details, including the repository name, contact information, and any additional setup steps or dependencies.
