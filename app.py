from flask import Flask, render_template, request, flash, redirect, url_for, session, send_file
from financial_twin import get_financial_twin, generic_ops
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend that works in any thread
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from urllib.parse import urlencode
import io
import zipfile
from sklearn.metrics import silhouette_score
import random
import webbrowser
from threading import Timer
import sys
import traceback

app = Flask(__name__)
app.secret_key = 'supersecretkey'


# ----------------------------
# Helpers
# ----------------------------

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS  # Set by PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def log_error(text: str):
    """Write errors to a local log file next to the exe (or cwd in dev)."""
    try:
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        with open(os.path.join(base, "app_error.log"), "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
    except Exception:
        pass


# ----------------------------
# Paths to packaged resources
# ----------------------------
drop_down_menu_file_path = resource_path("dependencies/drop_down_menu.xlsx")
shapefile_path = resource_path("dependencies/Shapes_NRW/dvg2gem_nw.shp")
fields_list_path = resource_path("dependencies/fields_list.xlsx")

# Check if the Excel file exists before attempting to read it
if not os.path.exists(drop_down_menu_file_path):
    raise FileNotFoundError(f"File not found: {drop_down_menu_file_path}")

try:
    municipalities_df = pd.read_excel(drop_down_menu_file_path, sheet_name="Municipalities")
    analysis_parameters_df = pd.read_excel(drop_down_menu_file_path, sheet_name="Parameters")
except Exception as e:
    raise RuntimeError(f"Error reading Excel file: {e}")

try:
    municipalities = municipalities_df['Municipality'].tolist()
    analysis_parameters = analysis_parameters_df['Parameter'].tolist()
except KeyError as e:
    raise KeyError(f"Column not found in Excel file: {e}")
except Exception as e:
    raise RuntimeError(f"Error processing Excel data: {e}")


# Initialize merged_df to None
merged_df = None


def initial_data(year=2019, include_per_capita='Yes'):
    global merged_df
    municipality = municipalities[0]
    parameter = analysis_parameters[0]
    _, merged_df = get_financial_twin(municipality, year, parameter, include_per_capita)


# Load initial data
initial_data()


# ----------------------------
# Routes
# ----------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        global merged_df

        default_year = 2019
        default_include_per_capita = 'Yes'

        year = ''

        if request.method == 'POST':
            year = int(request.form.get('year', default_year))
            include_per_capita = request.form.get('per_capita', default_include_per_capita)
            initial_data(year, include_per_capita)
        else:
            initial_data(default_year, default_include_per_capita)

        merged_df = merged_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # NOTE: This writes into current working directory; may fail under Program Files.
        # Keep as-is as per your current code, but consider writing to a user-writable folder later.
        merged_df.to_excel(f"mergedNRW_financialTwin_{year}.xlsx", index=False)

        if year == '':
            year = default_year
            include_per_capita = default_include_per_capita

        table_html = merged_df.to_html(
            classes='table table-striped',
            index=False,
            escape=False,
            formatters={
                'Payins Cosine Twin': lambda x: f'<a href="/analysis/{x}/Payins Cosine Twin/{year}/{include_per_capita}" class="btn btn-primary">{x}</a>',
                'Payins Euclidean Twin': lambda x: f'<a href="/analysis/{x}/Payins Euclidean Twin/{year}/{include_per_capita}" class="btn btn-primary">{x}</a>',
                'Payouts Cosine Twin': lambda x: f'<a href="/analysis/{x}/Payouts Cosine Twin/{year}/{include_per_capita}" class="btn btn-primary">{x}</a>',
                'Payouts Euclidean Twin': lambda x: f'<a href="/analysis/{x}/Payouts Euclidean Twin/{year}/{include_per_capita}" class="btn btn-primary">{x}</a>'
            }
        )
        table_html = table_html.replace('\n', '').replace('[', '').replace(']', '')
        return render_template('index.html', table_html=table_html, year=year, include_per_capita=include_per_capita)

    except Exception as e:
        flash(f"An error occurred while loading the page: {e}", 'error')
        return render_template('error.html'), 500


@app.route('/input')
def input_page():
    try:
        return render_template('input.html', municipalities=municipalities, parameters=analysis_parameters)
    except Exception as e:
        flash(f"An error occurred while loading the input page: {e}", 'error')
        return render_template('error.html'), 500


@app.route('/export_maps_data')
def export_maps_data():
    try:
        map_data = session.get('csv_files', [])

        if not map_data:
            flash("No maps data available for export.", "error")
            return redirect(url_for('plot_map'))

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for csv_filename in map_data:
                csv_path = os.path.join(app.static_folder, csv_filename)
                zip_file.write(csv_path, arcname=csv_filename)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name='map_data.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        flash(f"An error occurred while exporting the map data: {e}", "error")
        return redirect(url_for('plot_map'))


@app.route('/export_maps')
def export_maps():
    try:
        map_filenames = session.get('maps', [])

        if not map_filenames:
            flash("No maps available for export.", "error")
            return redirect(url_for('plot_map'))

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for map_filename in map_filenames:
                map_path = os.path.join(app.static_folder, map_filename)
                zip_file.write(map_path, arcname=map_filename)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name='maps.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        flash(f"An error occurred while exporting the maps: {e}", "error")
        return redirect(url_for('plot_map'))


@app.route('/find_twin', methods=['POST'])
def find_twin():
    try:
        municipality = request.form['municipality']
        year = int(request.form['year'])
        include_per_capita = request.form['per_capita']
        parameter = request.form['parameter']
    except KeyError as e:
        flash(f"Missing form data: {e}", 'error')
        return redirect(url_for('input_page'))

    try:
        global merged_df
        result, merged_df = get_financial_twin(municipality, year, parameter, include_per_capita)
        return render_template('result.html', result=result, municipality=municipality, method=parameter, year=year, include_per_capita=include_per_capita)
    except Exception as e:
        flash(f"An error occurred while finding the financial twin: {e}", 'error')
        return render_template('error.html'), 500


def find_closest_and_farthest_factors(municipality, financial_twin, financial_df, type, num_factors=5):
    try:
        financial_df = financial_df.loc[:, ~financial_df.columns.duplicated()]

        municipality_data = financial_df[financial_df['Gemeinde'] == municipality]
        twin_data = financial_df[financial_df['Gemeinde'] == financial_twin]

        if municipality_data.empty or twin_data.empty:
            raise ValueError("Municipality or financial twin data not found in the financial DataFrame.")

        all_columns = financial_df.columns.tolist()
        columns_to_compare = all_columns[7:]

        for col in columns_to_compare:
            financial_df[col] = pd.to_numeric(financial_df[col], errors='coerce')

        municipality_data = financial_df[financial_df['Gemeinde'] == municipality][columns_to_compare].astype(float)
        twin_data = financial_df[financial_df['Gemeinde'] == financial_twin][columns_to_compare].astype(float)

        differences = np.abs(municipality_data.values - twin_data.values).flatten()

        differences_df = pd.DataFrame(
            data={
                'Difference': differences,
                'Value_Municipality': municipality_data.values.flatten(),
                'Value_Twin': twin_data.values.flatten(),
            },
            index=columns_to_compare
        )

        closest_factors = differences_df.nsmallest(num_factors, 'Difference')
        farthest_factors = differences_df.nlargest(num_factors, 'Difference')

        return closest_factors, farthest_factors

    except Exception as e:
        raise RuntimeError(f"Error finding closest and farthest factors: {e}")


@app.route('/download_excel', methods=['GET'])
def download_excel():
    try:
        closest_factors = pd.DataFrame.from_dict(session['closest_factors'])
        farthest_factors = pd.DataFrame.from_dict(session['farthest_factors'])

        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        closest_factors.to_excel(writer, sheet_name='Closest Factors')
        farthest_factors.to_excel(writer, sheet_name='Farthest Factors')

        writer.close()
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name='analysis_factors.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        flash(f"An error occurred while generating the Excel file: {e}", 'error')
        return redirect(request.referrer)


@app.route('/analysis/<twin>/<type>/<year>/<per_capita>', methods=['GET', 'POST'])
def analysis(twin, type, year, per_capita):
    if request.method == 'POST':
        num_factors = int(request.form.get('num_factors', 5))
    else:
        num_factors = 5

    year = str(year)
    municipality = merged_df[merged_df[type] == twin]['Municipality'].values[0]
    merged_payins, merged_payouts, df_payins_relative, df_payouts_relative, _ = generic_ops()

    if 'Payins' in type and per_capita == 'No':
        financial_df = merged_payins
    elif 'Payins' in type and per_capita == 'Yes':
        financial_df = df_payins_relative
    elif 'Payouts' in type and per_capita == 'No':
        financial_df = merged_payouts
    elif 'Payouts' in type and per_capita == 'Yes':
        financial_df = df_payouts_relative

    financial_df = financial_df[financial_df['Year'] == year]

    closest_factors, farthest_factors = find_closest_and_farthest_factors(municipality, twin, financial_df, type, num_factors)

    session['closest_factors'] = closest_factors.to_dict()
    session['farthest_factors'] = farthest_factors.to_dict()

    return render_template(
        'analysis.html',
        municipality=municipality,
        twin=twin,
        closest_factors=closest_factors,
        farthest_factors=farthest_factors,
        num_factors=num_factors,
        year=year,
        include_per_capita=per_capita
    )


def generate_maps(financial_data_type, num_clusters, include_population, selected_fields):
    try:
        merged_payins, merged_payouts, df_payins_relative, df_payouts_relative, _ = generic_ops()

        if financial_data_type == 'Payins' and not include_population:
            final_df = merged_payins
            required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'EZ-Konto']
        elif financial_data_type == 'Payins' and include_population:
            final_df = df_payins_relative
            required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'EZ-Konto']
        elif financial_data_type == 'Payouts' and include_population:
            final_df = df_payouts_relative
            required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'AZ-Konto']
        elif financial_data_type == 'Payouts' and not include_population:
            final_df = merged_payouts
            required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'AZ-Konto']
        else:
            raise ValueError("Invalid financial data type specified")

        if include_population:
            selected_columns = [field + '_relative' for field in selected_fields]
        else:
            selected_columns = selected_fields

        selected_columns = required_columns + [col for col in selected_columns if col not in required_columns]
        selected_columns = [col.replace('\r\n', '\n') for col in selected_columns]
        final_df = final_df[selected_columns]

        gdf = gpd.read_file(shapefile_path)
        gdf.rename(columns={'GN': 'Municipality'}, inplace=True)

        final_df['Municipality'] = final_df['Gemeinde'].str.split(',').str[0]

        maps = []
        csv_files = []
        years = sorted(final_df['Year'].unique())

        # IMPORTANT: save to Flask's actual static folder so /static/... works everywhere
        temp_dir = app.static_folder
        os.makedirs(temp_dir, exist_ok=True)

        for year in years:
            yearly_df = final_df[final_df['Year'] == year]

            scaler = StandardScaler()
            all_columns = yearly_df.columns.tolist()
            columns_to_compare = all_columns[7:-1]

            normalized_data = scaler.fit_transform(yearly_df[columns_to_compare])

            num_clusters_int = int(num_clusters)
            kmeans = KMeans(n_clusters=num_clusters_int, random_state=0)
            cluster_labels = kmeans.fit_predict(normalized_data)

            yearly_df['Cluster'] = cluster_labels
            gdf_yearly = gdf.merge(yearly_df, left_on='Municipality', right_on='Municipality')

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            gdf_yearly.plot(column='Cluster', cmap='Set1', legend=True, ax=ax)

            for x, y, label in zip(
                gdf_yearly.geometry.centroid.x,
                gdf_yearly.geometry.centroid.y,
                gdf_yearly['Municipality']
            ):
                ax.text(x, y, label, fontsize=8)

            plt.title(f'Municipality Clusters in NRW - {year}')

            filename = f'map_{year}.png'
            plot_path = os.path.join(temp_dir, filename)

            plt.savefig(plot_path)
            plt.close()
            maps.append(filename)

            csv_filename = f'gdf_yearly_{year}.csv'
            csv_path = os.path.join(temp_dir, csv_filename)
            gdf_yearly.to_csv(csv_path, index=False)
            csv_files.append(csv_filename)

        return maps, csv_files, years

    except Exception:
        err = traceback.format_exc()
        print(err)
        log_error(err)
        return None, err


def normalize_data(df):
    scaler = StandardScaler()
    all_columns = df.columns.tolist()
    columns_to_compare = all_columns[7:-1]
    normalized_data = scaler.fit_transform(df[columns_to_compare])
    return normalized_data


def calculate_ideal_clusters(final_df, year_ideal):
    try:
        years = sorted(final_df['Year'].unique())
        total_clusters = 0
        ideal_num_clusters_year = None

        for year in years:
            yearly_df = final_df[final_df['Year'] == year]
            normalized_data = normalize_data(yearly_df)
            max_clusters = 10
            silhouette_scores = {}

            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                cluster_labels = kmeans.fit_predict(normalized_data)
                score = silhouette_score(normalized_data, cluster_labels)
                silhouette_scores[n_clusters] = score

            ideal_num_clusters = min(silhouette_scores, key=lambda k: abs(1 - silhouette_scores[k]))

            random_value = random.randint(3, 5)
            if ideal_num_clusters == 2:
                ideal_num_clusters = random_value

            if year_ideal is not None and int(year) == year_ideal:
                ideal_num_clusters_year = ideal_num_clusters

            total_clusters += ideal_num_clusters

        avg_ideal_clusters = total_clusters // len(years)
        return avg_ideal_clusters, ideal_num_clusters_year

    except Exception as e:
        raise RuntimeError(f"Error calculating ideal number of clusters: {e}")


@app.route('/plot_map', methods=['GET', 'POST'])
def plot_map():
    if request.method == 'POST':
        try:
            action = request.form.get('action')

            if action == 'calculate_clusters1' or action == 'calculate_clusters2':
                financial_data_type = request.form['financial_data_type']
                include_population = request.form.get('include_population') == 'on'
                selected_fields = request.form.getlist('fields')

                if action == 'calculate_clusters2':
                    year_ideal = int(request.form['year'])
                else:
                    year_ideal = None

                merged_payins, merged_payouts, df_payins_relative, df_payouts_relative, _ = generic_ops()

                if financial_data_type == 'Payins' and not include_population:
                    final_df = merged_payins
                    required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'EZ-Konto']
                elif financial_data_type == 'Payins' and include_population:
                    final_df = df_payins_relative
                    required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'EZ-Konto']
                elif financial_data_type == 'Payouts' and include_population:
                    final_df = df_payouts_relative
                    required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'AZ-Konto']
                elif financial_data_type == 'Payouts' and not include_population:
                    final_df = merged_payouts
                    required_columns = ['AGS_JuAmt', 'Bezeichnung2', 'Year', 'Population', 'AGS', 'Gemeinde', 'AZ-Konto']
                else:
                    raise ValueError("Invalid financial data type specified")

                if include_population:
                    selected_fields = [field + '_relative' for field in selected_fields]

                selected_columns = required_columns + [col for col in selected_fields if col not in required_columns]
                selected_columns = [col.replace('\r\n', '\n') for col in selected_columns]
                final_df = final_df[selected_columns]

                ideal_num_clusters, ideal_num_clusters_year = calculate_ideal_clusters(final_df, year_ideal)

                df = pd.read_excel(fields_list_path)
                fields = df['Fields'].tolist()
                return render_template('plot_map.html', fields=fields, num_clusters=ideal_num_clusters, ideal_clusters=ideal_num_clusters_year)

            else:
                financial_data_type = request.form['financial_data_type']
                num_clusters = request.form['num_clusters']
                include_population = request.form.get('include_population') == 'on'
                selected_fields = request.form.getlist('fields')

                result = generate_maps(financial_data_type, num_clusters, include_population, selected_fields)

                if result is None or result[0] is None:
                    err = result[1] if result else "Unknown error"
                    flash(f"Map generation failed:\n{err}", "error")
                    return render_template("error.html"), 500

                maps, csv_files, years = result
                session['maps'] = maps
                session['csv_files'] = csv_files

                query_params = {
                    'maps': maps,
                    'csv_files': csv_files,
                    'years': years,
                    'financial_data_type': financial_data_type,
                    'num_clusters': num_clusters,
                    'include_population': 'on' if include_population else 'off',
                    'fields': selected_fields
                }

                query_string = urlencode(query_params, doseq=True)
                return redirect(url_for('view_maps') + '?' + query_string)

        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            return render_template('error.html'), 500

    else:
        try:
            df = pd.read_excel(fields_list_path)
            fields = df['Fields'].tolist()
        except Exception as e:
            fields = []
            flash(f"An error occurred while loading the fields: {e}", 'error')

        return render_template('plot_map.html', fields=fields)


@app.route('/view_maps', methods=['GET'])
def view_maps():
    maps = request.args.getlist('maps')
    years = request.args.getlist('years')
    financial_data_type = request.args.get('financial_data_type', 'Payins')
    num_clusters = request.args.get('num_clusters', '4')
    include_population = request.args.get('include_population') == 'on'

    return render_template(
        'map_slider.html',
        maps=maps,
        years=years,
        enumerate=enumerate,
        financial_data_type=financial_data_type,
        num_clusters=num_clusters,
        include_population=include_population
    )


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == '__main__':
    try:
        Timer(1, open_browser).start()
        app.run()
    except Exception as e:
        print(f"An error occurred while running the app: {e}")
