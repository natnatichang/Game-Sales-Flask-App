# Description: A Flask web application to analyze video game sales data
# which includes historical sales analysis, regional distribution of games,
# and machine learning-based sales prediction for the future years

import io
import os
import sqlite3 as sl
import datetime
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

# Initialize Flask application
app = Flask(__name__)

# Prevent caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Database file name
DB_NAME = 'videogames.db'


# Initialize the SQLite databse with video game sales data from a csv file
# and cleans up the data to insert into a database table
def init_db():
    # Read the CSV file taken from https://www.kaggle.com/datasets/gregorut/videogamesales
    df = pd.read_csv('vgsales.csv')

    # Convert the columns to lowercase
    df.columns = df.columns.str.lower()

    # Handle the year column by filling NA values with 0, then convert to int
    df['year'] = df['year'].fillna(0).astype(int)

    # Connect to the database
    conn = sl.connect(DB_NAME)
    curs = conn.cursor()

    # Drop existing table if needed
    curs.execute('DROP TABLE IF EXISTS videogames')

    # Create a new table
    curs.execute('''CREATE TABLE IF NOT EXISTS videogames (
        rank INTEGER,
        name TEXT NOT NULL,
        platform TEXT NOT NULL,
        year INTEGER,
        genre TEXT,
        publisher TEXT,
        na_sales REAL,
        eu_sales REAL,
        jp_sales REAL,
        other_sales REAL,
        global_sales REAL
    )''')
    conn.commit()

    # Convert the dataframe to sql using pandas to do the "heavy lifting"
    df.to_sql('videogames', conn, if_exists='replace', index=False)

    # Close database connection
    conn.close()


# Get the top 10 games of the year to display on main page
def get_top_games():
    # Create connection to databse
    conn = sl.connect(DB_NAME)

    # Query to get the 10 top game name, platform released, year, and global sales from the database
    df = pd.read_sql_query("""
        SELECT name, platform, year, global_sales 
        FROM videogames 
        ORDER BY global_sales DESC 
        LIMIT 10
    """, conn)

    # Close database connection
    conn.close()

    # Convert to a list of dictionary before returning
    return df.to_dict('records')


# Gets a unique list of the platforms to show for dropdown with PC being first
def get_platforms():
    # Connect to the database
    conn = sl.connect(DB_NAME)

    # Get unique platform names from the database
    platforms_df = pd.read_sql_query("SELECT DISTINCT platform FROM videogames", conn)

    # Close database connection
    conn.close()

    platforms = platforms_df['platform'].tolist()

    # Remove PC from the list to put it in the front for easy access
    if 'PC' in platforms:
        platforms.remove('PC')

    # Insert into beginning
    platforms.insert(0, 'PC')

    # Return reorganized platform list
    return platforms


# Get available analysis options for the user to use
def get_analysis_options():
    return {
        "sales": "Historical Sales Analysis",
        "regional": "Regional Distribution",
        "prediction": "Future Sales Year Prediction",
    }


# Home page with GET route for the user and uses session management to handle default values
@app.route('/')
def home():
    # If there's no platform in session, default to PC
    if 'platform' not in session:
        session['platform'] = 'PC'

    # If there is no datarequest in session, default to sales
    if 'data_request' not in session:
        session['data_request'] = 'sales'

    # Return home page template with all the necessary data of top games list,
    # available platforms, analysis options, and the current session
    return render_template("home.html",
                           games=get_top_games(),
                           platforms=get_platforms(),
                           options=get_analysis_options(),
                           session=session)


# Preserve the session data with GET Routing based on what the user inputted for their choice of platform
@app.route("/back_to_home")
def back_to_home():
    # Only preserve platform and data_request
    platform = session.get('platform')
    data_request = session.get('data_request')

    # Clear the session
    session.clear()

    # Restore platform and data_request
    session['platform'] = platform
    session['data_request'] = data_request

    # Remove year/projection data
    if 'year' in session:
        session.pop('year')
    if 'projection_year' in session:
        session.pop('projection_year')

    # Redirect user back to home page
    return redirect(url_for('home'))


# Helps to submit form for analysis using POST endpoint
@app.route("/submit_analysis", methods=["POST"])
def submit_analysis():
    # Get user selected platform and analysis type from form and gives defaults in case
    platform = request.form.get("platform", "PS4")
    data_request = request.form.get("data_request", "sales")

    # Stores the selections into a session for persistence
    session["platform"] = platform
    session["data_request"] = data_request

    # Redirect to analysis page with parameters
    return redirect(url_for("show_analysis", data_request=data_request, platform=platform))


# Helps to implement the data visualization using GET dynamic routing
@app.route("/api/videogames/<data_request>/<platform>")
def show_analysis(data_request, platform):
    # Initialize prediction variable with NONE
    prediction = None

    # Check if it is a prediction request and there is a year
    if data_request == "prediction" and 'projection_year' in session:
        prediction = get_sales_prediction(platform, session['projection_year'])

    # Render the analysis template with all the data
    return render_template("locale.html",
                           data_request=data_request,
                           platform=platform,
                           prediction=prediction)


# Predicts the future gaming platform sales using Linear Regression
def get_sales_prediction(platform, projection_year):
    # Connects to database
    conn = sl.connect(DB_NAME)

    # Groups the data by years for total sales
    df = pd.read_sql_query("""
        SELECT 
            year,
            SUM(global_sales) as total_sales
        FROM videogames 
        WHERE platform = ? 
        AND year > 0
        AND year <= 2024
        GROUP BY year
        ORDER BY year
    """, conn, params=[platform])
    conn.close()

    # Convert year to datetime for consist processing
    df['year_date'] = pd.to_datetime(df['year'].astype(str), format='%Y')

    # Convert dates to ordinal numbers for ML since
    # ML needs numerical inputs
    df['yearmod'] = df['year_date'].map(datetime.datetime.toordinal)

    # Y - target variable for sales
    y = df['total_sales'].values

    # X - feature variable for years
    X = df['yearmod'].values.reshape(-1, 1)

    # Convert projection year to same format as the data
    projection_date = datetime.datetime(projection_year, 1, 1)
    projection_ordinal = datetime.datetime.toordinal(projection_date)

    # Create the linear regression model
    regr = LinearRegression(fit_intercept=True)

    # Train the model based on the x, y historical data
    regr.fit(X, y)

    # Make prediction for future years and includes 2d array for sklearn
    pred = float(regr.predict([[projection_ordinal]])[0])

    # Returns the prediction
    return pred


# Generate and show different types of visualizations based on request with GET dynamic routing
@app.route("/fig/<data_request>/<platform>")
def fig(data_request, platform):
    # Prediction visualization for the line chart if year was set in the session
    if data_request == "prediction" and 'year' in session:
        # Only show prediction if we have explicitly set a year through the form
        projection_year = session.get('year')
        fig = create_sales_figure(platform, projection_year, show_prediction=True)

    # Shows the historical line chart data of the past as is
    elif data_request == "sales":
        fig = create_sales_figure(platform, show_prediction=False)

    # Shows the pie chart with sales distribution across regions
    elif data_request == "regional":
        fig = create_regional_figure(platform)

    # Default to showing historical data just in case
    else:
        fig = create_sales_figure(platform, show_prediction=False)

    # Create in-memory buffer for image
    img = io.BytesIO()

    # Save the figure to buffer
    fig.savefig(img, format='png', bbox_inches='tight')

    # Reset buffer position
    img.seek(0)

    # Send back image with correct type
    return send_file(img, mimetype="image/png")


# Creates a visualization of the historical sales data with the option to see future predictions
def create_sales_figure(platform, projection_year=None, show_prediction=False):
    # Create connection to database
    conn = sl.connect(DB_NAME)

    # Get the historical data from the database
    df = pd.read_sql_query("""
        SELECT 
            year,
            SUM(global_sales) as total_sales
        FROM videogames 
        WHERE platform = ? 
        AND year > 0
        AND year <= 2024
        GROUP BY year
        ORDER BY year
    """, conn, params=[platform])
    conn.close()

    # Convert years to datatime
    df['year_date'] = pd.to_datetime(df['year'].astype(str), format='%Y')

    # Create figure with chosen size
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Check if show_prediction and projection_years are both fulfilled
    if show_prediction and projection_year:
        # Get sales prediction for the future years
        predicted_sales = get_sales_prediction(platform, projection_year)

        # Create a single point DataFrame for said prediction
        pred_date = datetime.datetime(projection_year, 1, 1)
        df_pred = pd.DataFrame({
            'year_date': [pred_date],
            'total_sales': [predicted_sales]
        })

        # Plot historical data
        ax.plot(df['year_date'],
                df['total_sales'],
                color='#6a0dad',
                marker='o',
                label='Historical Sales')

        # Plot prediction line that connects to last point of the original data
        ax.plot([df['year_date'].iloc[-1], pred_date],
                [df['total_sales'].iloc[-1], predicted_sales],
                color='#9370db',
                linestyle='--',
                linewidth=2,
                label='Predicted Sales')

        # Plot prediction points
        ax.scatter([pred_date], [predicted_sales],
                   color='#9370db',
                   s=100)

        # Set title for prediction view
        ax.set_title(f'Predictive Sales Analysis for {platform} through {projection_year}')
    else:
        # Plot only historical date if there were no predictions for future years requested
        ax.plot(df['year_date'],
                df['total_sales'],
                color='#6a0dad',
                marker='o',
                label='Historical Sales')

        # Set title for historical view
        ax.set_title(f'Historical Sales Data for {platform}')

    # Add axis labels and a legend
    ax.set_xlabel('Year')
    ax.set_ylabel('Global Sales (millions)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis to show years for easy reading
    ax.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent cutoff
    fig.tight_layout()

    # Return the figure back so the user can see
    return fig


# Creates line chart visualization for historical sales data for the gaming platform of choice
def create_prediction_figure(platform):
    # Creates a connection to the database
    conn = sl.connect(DB_NAME)

    # Use the SUm and COUNT aggregation to group the data and filter
    df = pd.read_sql_query("""
        SELECT year, SUM(global_sales) as total_sales, COUNT(*) as game_count
        FROM videogames 
        WHERE platform = ? 
        AND year NOT IN (2017, 2020)
        AND year > 0  
        AND year <= 2024 
        GROUP BY year
        HAVING game_count >= 5
        ORDER BY year
    """, conn, params=[platform])
    conn.close()

    # Remove any rows that are NA just in case
    df = df.dropna()

    # Create figure with specific sizes
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Create the line chart
    ax.bar(df['year'], df['total_sales'], color='#C8B4E2')

    # Title the chart
    ax.set_title('Historical Sales Analysis for ' + str(platform))

    # Add chart axis labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Global Sales (millions)')

    # Rotate axis for easier reading
    ax.tick_params(axis='x', rotation=45)

    # Make sure the range is correct
    if not df.empty:
        ax.set_xlim(df['year'].min() - 1, df['year'].max() + 1)

    # Return the figure back to display to the user
    return fig


# Creates a pie chart for the regional sales distribution
def create_regional_figure(platform):
    # Connects to the database
    conn = sl.connect(DB_NAME)

     # Uses SQl aggregation SUM() to aggregate data across regions
    df = pd.read_sql_query("""
        SELECT 
            SUM(NA_Sales) as NA_Sales,
            SUM(EU_Sales) as EU_Sales,
            SUM(JP_Sales) as JP_Sales,
            SUM(Other_Sales) as Other_Sales
        FROM videogames 
        WHERE Platform = ? 
        AND Year NOT IN (2017, 2020)
    """, conn, params=[platform])

    # Close the connection
    conn.close()

    # Custom colors for each slice of the pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    # Creates a piechart showing the regional sales distribution
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Defines the labels and data size for the part chart
    labels = ['North America', 'Europe', 'Japan', 'Other Regions']
    sizes = [df['NA_Sales'][0], df['EU_Sales'][0],
             df['JP_Sales'][0], df['Other_Sales'][0]]

    # Create pie chart with the percentage labels
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)

    # Shows the legend and title of the pie chart
    ax.legend()
    ax.set_title('Regional Sales Distribution for ' + platform)

    # Return the figure back to display
    return fig


# Handles project form submission with a POST endpoint to process form data
@app.route("/submit_projection", methods=['POST'])
def submit_projection():
    # Verify platform exists in session for security
    if 'platform' not in session:
        return redirect(url_for("home"))

    # Stores the submitted year in the session after converting to ensure it is int
    session["year"] = int(request.form["year"])

    # Redirect the projection display route
    return redirect(url_for("show_projection",
                            platform=session["platform"],
                            year=session["year"]))


# Displays video game sales projection results using GET dynamic route
@app.route("/api/videogames/projection/<platform>/<year>")
def show_projection(platform, year):
    # Convert year parameter to int for calculations
    year = int(year)

    # Get the sales prediction using ML model
    pred_sales = get_sales_prediction(platform, year)

    # Return the template with the data
    return render_template("projection.html",
                           platform=platform,
                           year=year,
                           prediction=pred_sales)


# Redirects all undefined routes back to home with GET dynamic route
@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home"))


# Main to run the database creation and manage the Flask website
if __name__ == "__main__":

    if not os.path.exists(DB_NAME):

        init_db()
    else:

        conn = sl.connect(DB_NAME)
        count = pd.read_sql_query("SELECT COUNT(*) as count FROM videogames", conn).iloc[0]['count']
        conn.close()
        if count == 0:
            init_db()

    app.secret_key = os.urandom(12)
    app.run(debug=True)
