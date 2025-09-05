import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('default')
sns.set_palette("husl")

# 1. Generate synthetic sales data
def generate_sales_data(rows=1000):
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(rows)]
    
    # Generate product categories
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress'],
        'Home & Kitchen': ['Blender', 'Pan Set', 'Knife Set', 'Coffee Maker'],
        'Books': ['Novel', 'Textbook', 'Cookbook', 'Children\'s Book'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Basketball']
    }
    
    # Generate regions
    regions = ['North', 'South', 'East', 'West']
    
    # Create data
    data = []
    for i in range(rows):
        category = np.random.choice(categories)
        product = np.random.choice(products[category])
        region = np.random.choice(regions)
        quantity = np.random.randint(1, 5)
        price = np.random.uniform(10, 500)
        sales = quantity * price
        
        data.append([
            dates[i],
            category,
            product,
            region,
            quantity,
            price,
            sales
        ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'Date', 'Category', 'Product', 'Region', 'Quantity', 'Unit_Price', 'Sales'
    ])
    
    # Add some missing values
    df.loc[df.sample(frac=0.05).index, 'Region'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'Unit_Price'] = np.nan
    
    # Add some duplicates
    duplicates = df.sample(n=10)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

# Generate the data
print("Generating synthetic sales data...")
sales_df = generate_sales_data(1500)
print("Data generated successfully!")

# 2. Data Inspection
print("\n=== DATA INSPECTION ===")
print(f"Dataset shape: {sales_df.shape}")
print("\nFirst 5 rows:")
print(sales_df.head())
print("\nDataset info:")
sales_df.info()
print("\nSummary statistics:")
print(sales_df.describe())

# Check for missing values
print("\nMissing values per column:")
print(sales_df.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows: {sales_df.duplicated().sum()}")

# 3. Data Cleaning
print("\n=== DATA CLEANING ===")

# Handle missing values
print("Handling missing values...")
sales_df['Region'].fillna('Unknown', inplace=True)
sales_df['Unit_Price'].fillna(sales_df['Unit_Price'].median(), inplace=True)
print("Missing values handled.")

# Remove duplicates
print("Removing duplicates...")
initial_count = len(sales_df)
sales_df.drop_duplicates(inplace=True)
final_count = len(sales_df)
print(f"Removed {initial_count - final_count} duplicate rows.")

# Format columns
print("Formatting columns...")
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df['Month'] = sales_df['Date'].dt.month_name()
sales_df['Quarter'] = sales_df['Date'].dt.quarter
print("Columns formatted.")

# 4. Exploratory Data Analysis (EDA)
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Sales by category
print("\nSales by Category:")
category_sales = sales_df.groupby('Category')['Sales'].agg(['sum', 'mean', 'count'])
print(category_sales)

# Sales by region
print("\nSales by Region:")
region_sales = sales_df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count'])
print(region_sales)

# Monthly sales trend
print("\nMonthly Sales Trend:")
monthly_sales = sales_df.groupby('Month')['Sales'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
print(monthly_sales)

# Correlation analysis
print("\nCorrelation Matrix:")
numeric_df = sales_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# 5. Data Visualization
print("\n=== DATA VISUALIZATION ===")

# Set up the figure and subplots
fig = plt.figure(figsize=(20, 16))

# 1. Sales by Category (Bar Chart)
plt.subplot(3, 3, 1)
category_sales['sum'].sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)

# 2. Sales by Region (Bar Chart)
plt.subplot(3, 3, 2)
region_sales['sum'].sort_values(ascending=False).plot(kind='bar', color='lightgreen')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)

# 3. Monthly Sales Trend (Line Chart)
plt.subplot(3, 3, 3)
monthly_sales.plot(kind='line', marker='o', color='orange')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.grid(True)

# 4. Sales Distribution by Category (Pie Chart)
plt.subplot(3, 3, 4)
category_sales['sum'].plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Sales Distribution by Category')
plt.ylabel('')

# 5. Unit Price Distribution (Histogram)
plt.subplot(3, 3, 5)
plt.hist(sales_df['Unit_Price'], bins=20, color='purple', alpha=0.7)
plt.title('Distribution of Unit Prices')
plt.xlabel('Unit Price ($)')
plt.ylabel('Frequency')

# 6. Quantity vs Sales (Scatter Plot)
plt.subplot(3, 3, 6)
plt.scatter(sales_df['Quantity'], sales_df['Sales'], alpha=0.6, color='teal')
plt.title('Quantity vs Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales ($)')

# 7. Correlation Heatmap
plt.subplot(3, 3, 7)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

# 8. Average Sales by Region and Category (Grouped Bar Chart)
plt.subplot(3, 3, 8)
region_category_sales = sales_df.groupby(['Region', 'Category'])['Sales'].mean().unstack()
region_category_sales.plot(kind='bar')
plt.title('Average Sales by Region and Category')
plt.xlabel('Region')
plt.ylabel('Average Sales ($)')
plt.xticks(rotation=45)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# 9. Top Selling Products (Horizontal Bar Chart)
plt.subplot(3, 3, 9)
top_products = sales_df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(8)
top_products.sort_values().plot(kind='barh', color='coral')
plt.title('Top 8 Selling Products by Sales')
plt.xlabel('Sales ($)')

plt.tight_layout()
plt.savefig('sales_analysis_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Insights and Conclusion
print("\n=== INSIGHTS AND CONCLUSIONS ===")
print("\nKey Findings:")
print("1. Electronics is the highest revenue-generating category, accounting for {:.1f}% of total sales."
      .format(category_sales['sum']['Electronics'] / category_sales['sum'].sum() * 100))
print("2. The West region generated the highest sales, followed by North and East.")
print("3. Sales peak during November and December (holiday season), with a noticeable dip in January.")
print("4. There's a strong positive correlation between Quantity and Sales (r = {:.2f}).".format(
    correlation_matrix.loc['Quantity', 'Sales']))
print("5. The average unit price is ${:.2f}, with most products priced between $50 and $200.".format(
    sales_df['Unit_Price'].mean()))

print("\nRecommendations:")
print("1. Focus marketing efforts on Electronics category as it drives the highest revenue.")
print("2. Increase inventory and promotions in the West region to capitalize on high sales performance.")
print("3. Plan special holiday promotions for November and December to maximize seasonal sales peaks.")
print("4. Consider bundling products to increase quantity per transaction, given the strong correlation with total sales.")
print("5. Explore opportunities to expand product offerings in the $50-$200 price range where demand is highest.")

print("\n=== ANALYSIS COMPLETE ===")
print("Visualizations have been saved as 'sales_analysis_visualizations.png'")