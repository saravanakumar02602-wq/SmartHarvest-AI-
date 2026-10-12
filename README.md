# SmartHarvest AI

SmartHarvest AI is a local R Shiny application for managing farm operations and reviewing crop, finance, harvest, and soil information.

## Features

- Dashboard with cultivated land, sales revenue, estimated net profit, expense breakdown, and expected production.
- Crop, expense, harvest, sales, and soil record management.
- Crop yield prediction using a random forest model trained from historical yield data.
- Educational, rule-based crop disease symptom guidance. This is a demonstration, not an image-recognition diagnosis.
- CSV downloads and farm reports.

## Requirements

- R for Windows or another supported operating system.
- The following R packages:

```r
install.packages(c(
  "shiny", "bslib", "DT", "dplyr", "ggplot2",
  "plotly", "randomForest", "base64enc"
))
```

Run the installation command in the R or RStudio Console, not directly in PowerShell.

## Run the application

### RStudio

1. Open `Agriculture_Management_System.Rproj` in RStudio.
2. Open `app.R`.
3. Click **Run App**.

### R command line

Start R from this project directory and run:

```r
shiny::runApp()
```

Alternatively, from PowerShell with R installed at the usual Windows location:

```powershell
Set-Location "C:\path\to\Agriculture_Management_System"
& "C:\Program Files\R\R-4.6.1\bin\R.exe" --vanilla -e "shiny::runApp('.', launch.browser=TRUE)"
```

If R is installed in another folder, replace the executable path with the installed R version's `R.exe` path.

Shiny prints a local URL (usually `http://127.0.0.1:<port>`) when the application starts. Open that URL in a browser.

## Data

The application reads and writes its CSV records in the `data` folder relative to the project directory. Keep `app.R` and `data` together, and launch the application from the project directory so it uses the intended data files.
