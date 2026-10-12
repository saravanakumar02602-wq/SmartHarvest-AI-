# ==============================================================================
# PROJECT: Smart Agriculture Management System
# PHASE 8: ML Crop Yield Prediction (Phase 7 Weather Removed)
# TECHNOLOGIES: R, Shiny, bslib, DT, dplyr, ggplot2, plotly, randomForest
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. LOAD LIBRARIES
# ------------------------------------------------------------------------------
library(shiny)
library(bslib)
library(DT)
library(dplyr)
library(ggplot2)
library(plotly)
library(randomForest)
library(base64enc)

# ------------------------------------------------------------------------------
# 2. DATA INITIALIZATION HELPER
# ------------------------------------------------------------------------------
data_dir <- "data"
crops_csv    <- file.path(data_dir, "crops.csv")
expenses_csv <- file.path(data_dir, "expenses.csv")
sales_csv    <- file.path(data_dir, "sales.csv")
harvest_csv  <- file.path(data_dir, "harvest.csv")
soil_csv     <- file.path(data_dir, "soil.csv")
yield_csv    <- file.path(data_dir, "historical_yield.csv")

if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)

if (!file.exists(crops_csv)) {
  write.csv(data.frame(
    Crop_ID = c("CROP001","CROP002","CROP003","CROP004","CROP005"),
    Crop_Name = c("Wheat","Rice","Cotton","Maize","Tomato"),
    Variety = c("HD-2967","Basmati 370","Bt Cotton","African Tall","Roma VF"),
    Area_Acres = c(12.5, 20.0, 8.0, 15.0, 4.5),
    Sowing_Date = c("2026-05-15","2026-06-10","2026-04-25","2026-07-01","2026-08-01"),
    Expected_Harvest_Date = c("2026-10-20","2026-11-15","2026-10-30","2026-10-15","2026-11-30"),
    Status = c("Growing","Growing","Growing","Growing","Growing"),
    Yield_Per_Acre_Kg = c(1800, 2200, 950, 2400, 3200),
    stringsAsFactors = FALSE
  ), crops_csv, row.names = FALSE)
}

if (!file.exists(expenses_csv)) {
  write.csv(data.frame(
    Expense_ID = c("EXP001","EXP002","EXP003","EXP004","EXP005","EXP006","EXP007","EXP008","EXP009","EXP010","EXP011"),
    Category = c("Seeds","Seeds","Seeds","Fertilizers","Fertilizers","Labor","Irrigation","Pesticides","Equipment","Seeds","Seeds"),
    Description = c("Certified Wheat Seeds","Basmati 370 Seeds","Bt Cotton Hybrid Seeds","Urea & DAP Fertilizer","Potash & Micronutrients","Field Preparation & Labor","Canal Water & Drip","Neem Oil Pest Spray","Tractor Rental & Fuel","Tomato Hybrid Seedlings","Fodder Maize Seeds"),
    Amount = c(15000, 22000, 18000, 45000, 25000, 55000, 28000, 20000, 35000, 8000, 12000),
    Date = c("2026-05-10","2026-06-05","2026-04-20","2026-06-15","2026-06-25","2026-05-18","2026-07-05","2026-07-20","2026-05-02","2026-07-28","2026-06-28"),
    Crop_ID = c("CROP001","CROP002","CROP003","All","All","All","All","CROP003","All","CROP005","CROP004"),
    stringsAsFactors = FALSE
  ), expenses_csv, row.names = FALSE)
}

if (!file.exists(sales_csv)) {
  write.csv(data.frame(
    Sale_ID = c("SALE001","SALE002","SALE003"),
    Crop_ID = c("CROP002","CROP001","CROP005"),
    Crop_Name = c("Rice","Wheat","Tomato"),
    Sale_Date = c("2026-07-15","2026-07-25","2026-08-10"),
    Quantity_Kg = c(8000, 6000, 3500),
    Unit_Price = c(32.0, 26.0, 20.0),
    Total_Revenue = c(256000, 156000, 70000),
    stringsAsFactors = FALSE
  ), sales_csv, row.names = FALSE)
}

if (!file.exists(harvest_csv)) {
  write.csv(data.frame(
    Harvest_ID = c("HARV001","HARV002","HARV003"),
    Crop_ID = c("CROP001","CROP002","CROP005"),
    Crop_Name = c("Wheat","Rice","Tomato"),
    Harvest_Date = c("2026-06-28","2026-07-10","2026-08-05"),
    Quantity_Kg = c(22500, 44000, 14400),
    Yield_Per_Acre_Kg = c(1800, 2200, 3200),
    Quality_Grade = c("Grade A","Grade A+","Grade A"),
    stringsAsFactors = FALSE
  ), harvest_csv, row.names = FALSE)
}

if (!file.exists(soil_csv)) {
  write.csv(data.frame(
    Soil_ID = c("SOIL001","SOIL002","SOIL003","SOIL004","SOIL005"),
    Field_Name = c("North Block - Parcel A","South Basin - Lowland","East Terrace - Parcel C","West Flat - Parcel D","Central Plot - Greenhouse"),
    Linked_Crop_ID = c("CROP001","CROP002","CROP003","CROP004","CROP005"),
    Soil_Type = c("Clay Loam","Alluvial","Black Cotton","Sandy Loam","Red Sandy Loam"),
    pH = c(6.8, 7.2, 8.1, 6.4, 6.9),
    Nitrogen_kg_ha = c(280, 240, 190, 310, 260),
    Phosphorus_kg_ha = c(38, 28, 22, 45, 34),
    Potassium_kg_ha = c(220, 180, 160, 250, 210),
    Moisture_Percent = c(24.5, 32.0, 18.2, 22.0, 26.5),
    Test_Date = c("2026-05-12","2026-06-08","2026-04-22","2026-06-25","2026-07-28"),
    Health_Status = c("Optimal","Optimal","Alkaline","Slightly Acidic","Optimal"),
    stringsAsFactors = FALSE
  ), soil_csv, row.names = FALSE)
}

# Generate historical yield training data if it doesn't exist
if (!file.exists(yield_csv)) {
  set.seed(42)
  crops_list <- c("Wheat","Rice","Cotton","Maize","Tomato","Soybean","Potato")
  n_records  <- 280
  records    <- list()

  crop_params <- list(
    Wheat   = list(ph=6.6, n=260, p=38,  k=210, rain=550,  temp=23, area_lo=4,  area_hi=30, base=2100),
    Rice    = list(ph=6.2, n=290, p=32,  k=230, rain=1150, temp=28, area_lo=5,  area_hi=40, base=2500),
    Cotton  = list(ph=7.1, n=210, p=28,  k=180, rain=680,  temp=30, area_lo=6,  area_hi=35, base=1050),
    Maize   = list(ph=6.5, n=280, p=42,  k=220, rain=750,  temp=26, area_lo=3,  area_hi=25, base=2450),
    Tomato  = list(ph=6.4, n=250, p=48,  k=260, rain=620,  temp=24, area_lo=2,  area_hi=15, base=3400),
    Soybean = list(ph=6.5, n=180, p=40,  k=190, rain=720,  temp=25, area_lo=5,  area_hi=30, base=1450),
    Potato  = list(ph=5.8, n=270, p=55,  k=290, rain=580,  temp=20, area_lo=3,  area_hi=20, base=4300)
  )

  for (i in 1:n_records) {
    cr  <- sample(crops_list, 1)
    p   <- crop_params[[cr]]
    ph  <- round(max(4.8, min(8.8, rnorm(1, p$ph,  0.4))), 2)
    n_v <- round(max(80,  min(420, rnorm(1, p$n,   30))))
    p_v <- round(max(10,  min(85,  rnorm(1, p$p,   6))))
    k_v <- round(max(80,  min(380, rnorm(1, p$k,   25))))
    rain<- round(max(200, min(1600,rnorm(1, p$rain, 90))))
    temp<- round(max(14,  min(42,  rnorm(1, p$temp, 2.5))), 1)
    area<- round(runif(1, p$area_lo, p$area_hi), 1)

    yield <- round(max(400,
      p$base
      - 150*(ph   - p$ph)^2
      - 35 *(temp - p$temp)^2
      - 0.7*(rain - p$rain)^2/100
      + 0.7*(n_v  - p$n*0.85)
      + 2.5*(p_v  - p$p*0.8)
      + 0.8*(k_v  - p$k*0.85)
      + rnorm(1, 0, 55)
    ))

    records[[i]] <- data.frame(
      Record_ID           = sprintf("HY%03d", i),
      Crop                = cr,
      Area_Acres          = area,
      Soil_pH             = ph,
      Nitrogen_kg_ha      = n_v,
      Phosphorus_kg_ha    = p_v,
      Potassium_kg_ha     = k_v,
      Rainfall_mm         = rain,
      Temperature_C       = temp,
      Actual_Yield_kg_per_acre = yield,
      stringsAsFactors    = FALSE
    )
  }
  write.csv(do.call(rbind, records), yield_csv, row.names = FALSE)
}

# Train ML model at startup (global, so it is built once)
train_rf_model <- function() {
  df <- read.csv(yield_csv, stringsAsFactors = FALSE)
  df$Crop <- as.factor(df$Crop)
  set.seed(123)
  idx   <- sample(nrow(df), floor(0.8 * nrow(df)))
  train <- df[idx, ]
  test  <- df[-idx, ]

  features <- c("Crop","Area_Acres","Soil_pH","Nitrogen_kg_ha",
                 "Phosphorus_kg_ha","Potassium_kg_ha","Rainfall_mm","Temperature_C")

  rf_model <- randomForest(
    x         = train[, features],
    y         = train$Actual_Yield_kg_per_acre,
    ntree     = 300,
    importance= TRUE
  )

  preds <- predict(rf_model, newdata = test[, features])
  rmse  <- round(sqrt(mean((preds - test$Actual_Yield_kg_per_acre)^2)), 1)
  ss_res<- sum((test$Actual_Yield_kg_per_acre - preds)^2)
  ss_tot<- sum((test$Actual_Yield_kg_per_acre - mean(test$Actual_Yield_kg_per_acre))^2)
  r2    <- round(1 - ss_res/ss_tot, 3)

  list(model = rf_model, rmse = rmse, r2 = r2,
       test_actual = test$Actual_Yield_kg_per_acre, test_preds = preds,
       test_crop = test$Crop, n_train = nrow(train), n_test = nrow(test))
}

ml_result <- train_rf_model()

# ------------------------------------------------------------------------------
# 3. USER INTERFACE (UI) DEFINITION
# ------------------------------------------------------------------------------
ui <- page_navbar(
  title = div(
    class = "d-flex align-items-center gap-2",
    span("🌱", style = "font-size: 1.6rem;"),
    div(
      div(class = "fw-bold fs-5 text-white lh-1", "SmartHarvest AI"),
      tags$small(class = "text-white-50", "Precision Agriculture Management Platform", style = "font-size: 0.72rem;")
    )
  ),
  theme = bs_theme(
    version    = 5,
    bootswatch = "flatly",
    primary    = "#1b5e20",
    secondary  = "#2e7d32",
    success    = "#2e7d32",
    warning    = "#f57f17",
    danger     = "#d32f2f",
    base_font  = font_google("Plus Jakarta Sans")
  ),
  fillable = FALSE,

  # Custom CSS
  header = tags$head(
    tags$style(HTML("
      body { background-color: #f4f7f5; font-family: 'Plus Jakarta Sans', sans-serif; }
      .navbar {
        background: linear-gradient(135deg, #17324b 0%, #245b65 60%, #397b70 100%) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        padding-top: 0.6rem; padding-bottom: 0.6rem;
      }
      .navbar-nav .nav-link { color: rgba(255,255,255,0.85) !important; font-weight: 500; font-size: 0.84rem; padding: 0.45rem 0.45rem !important; border-radius: 8px; transition: all 0.2s ease; }
      .navbar-nav .nav-link:hover, .navbar-nav .nav-link.active { color: #ffffff !important; background: rgba(255,255,255,0.15) !important; }
      .hero-card { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%); color: white; border-radius: 18px; padding: 26px 32px; box-shadow: 0 8px 25px rgba(27,94,32,0.2); margin-bottom: 22px; }
      .dashboard-page { background: #f0eee7; border: 1px solid #e4e0d5; border-radius: 24px; padding: clamp(12px, 2vw, 26px); }
      .dashboard-hero { position: relative; overflow: hidden; background: #202d42; border-left: 7px solid #d8a84e; border-radius: 12px; padding: clamp(24px, 4vw, 42px); color: #fff; box-shadow: 0 12px 28px rgba(32,45,66,0.14); margin-bottom: 18px; }
      .dashboard-hero:after { content: ''; position: absolute; width: 210px; height: 210px; border: 1px solid rgba(233,190,105,0.22); border-radius: 50%; right: 7%; top: -145px; box-shadow: 0 0 0 22px rgba(233,190,105,0.04), 0 0 0 44px rgba(233,190,105,0.035); pointer-events: none; }
      .dashboard-eyebrow { color: #e9be69; font-size: 0.76rem; font-weight: 800; letter-spacing: 1.7px; text-transform: uppercase; }
      .dashboard-hero h1 { font-size: clamp(1.75rem, 3vw, 2.65rem); letter-spacing: -1.2px; line-height: 1.12; }
      .dashboard-hero-copy { max-width: 590px; color: rgba(255,255,255,0.82); line-height: 1.65; }
      .dashboard-section-heading { color: #20334d; font-size: 1.12rem; font-weight: 800; letter-spacing: -0.3px; }
      .dashboard-section-note { color: #738078; font-size: 0.84rem; }
      .dashboard-stat-card { height: 100%; min-height: 170px; background: #fff; border: 1px solid #e5e1d7; border-top: 4px solid #587b84; border-radius: 10px; padding: 20px; box-shadow: 0 5px 15px rgba(32,45,66,0.045); transition: transform 0.18s ease, box-shadow 0.18s ease; }
      .dashboard-stat-card:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(32,45,66,0.09); }
      .dashboard-stat-card:nth-child(2) { border-top-color: #cf815f; }
      .dashboard-stat-card:nth-child(3) { border-top-color: #d8a84e; }
      .dashboard-stat-label { color: #627168; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; }
      .dashboard-stat-value { color: #173b2b; font-size: clamp(1.45rem, 2.2vw, 1.9rem); font-weight: 800; letter-spacing: -0.7px; line-height: 1.2; overflow-wrap: anywhere; }
      .dashboard-stat-note { color: #76827a; font-size: 0.82rem; }
      .dashboard-stat-icon { width: 44px; height: 44px; flex: 0 0 44px; display: grid; place-items: center; border-radius: 13px; font-size: 1.3rem; }
      .dashboard-stat-card:nth-child(1) .dashboard-stat-icon { background: #e4f2ee !important; color: #397b70 !important; }
      .dashboard-stat-card:nth-child(2) .dashboard-stat-icon { background: #f6e8e2 !important; color: #b96348 !important; }
      .dashboard-stat-card:nth-child(3) .dashboard-stat-icon { background: #fbf0d8 !important; color: #ad7820 !important; }
      .dashboard-land-progress { height: 7px; background: #e9f0eb; border-radius: 8px; overflow: hidden; }
      .dashboard-land-progress .progress-bar { background: linear-gradient(90deg, #397b70, #7fb38b); border-radius: 8px; }
      .dashboard-chart-card { height: 100%; background: #fff; border: 1px solid #e5e1d7; border-radius: 12px; box-shadow: 0 5px 15px rgba(32,45,66,0.045); overflow: hidden; }
      .dashboard-chart-header { padding: 18px 21px 0; }
      .dashboard-chart-title { color: #20334d; font-size: 0.98rem; font-weight: 800; }
      .dashboard-chart-note { color: #7a867f; font-size: 0.78rem; margin-top: 3px; }
      .dashboard-chart-body { padding: 8px 12px 12px; }
      .dashboard-toolbar { background: #fff; border: 1px solid #e7eee9; border-radius: 15px; padding: 14px 18px; box-shadow: 0 4px 14px rgba(24,57,38,0.035); }
      .dashboard-toolbar .btn { min-height: 40px; border-radius: 10px; }
      .dashboard-page .btn-success { background: #bf704f; border-color: #bf704f; }
      .dashboard-page .btn-success:hover { background: #a95e40; border-color: #a95e40; }
      @media (max-width: 575.98px) {
        .dashboard-hero { border-radius: 17px; padding: 23px 20px; }
        .dashboard-stat-card { min-height: 164px; padding: 16px; }
        .dashboard-chart-header { padding: 16px 16px 0; }
      }
      .ml-hero-card { background: linear-gradient(135deg, #1565c0 0%, #1976d2 60%, #42a5f5 100%); color: white; border-radius: 18px; padding: 26px 32px; box-shadow: 0 8px 25px rgba(21,101,192,0.2); margin-bottom: 22px; }
      .kpi-container { background: #ffffff; border-radius: 16px; padding: 20px 22px; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 4px 20px rgba(0,0,0,0.04); transition: transform 0.2s ease, box-shadow 0.2s ease; }
      .kpi-container:hover { transform: translateY(-4px); box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
      .kpi-icon-bubble { width: 52px; height: 52px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; }
      .kpi-title { color: #556958; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; margin-bottom: 4px; }
      .kpi-value { font-size: 1.9rem; font-weight: 800; color: #1a2e1e; line-height: 1.1; }
      .kpi-subtext { font-size: 0.8rem; margin-top: 6px; }
      .card-modern { background: #ffffff; border-radius: 16px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 20px rgba(0,0,0,0.04); margin-bottom: 22px; overflow: hidden; }
      .card-modern .card-header { background: #ffffff; border-bottom: 1px solid #edf2ee; padding: 16px 22px; font-weight: 700; color: #1b3820; display: flex; align-items: center; justify-content: space-between; }
      .advisory-box { background: #f8faf8; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
      .ml-advisory-box { background: #e3f2fd; border-left: 4px solid #1565c0; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
      .soil-advisory-alert { background: #fffde7; border-left: 4px solid #fbc02d; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; }
      .filter-toolbar { background: #ffffff; border-radius: 14px; padding: 14px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); border: 1px solid rgba(0,0,0,0.05); margin-bottom: 22px; }
      .badge-soft-success { background: #e8f5e9; color: #2e7d32; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
      .badge-soft-info    { background: #e1f5fe; color: #0277bd; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
      .badge-soft-warning { background: #fff8e1; color: #f57f17; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
      .badge-soft-danger  { background: #ffebee; color: #c62828; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
      .badge-soft-primary { background: #e3f2fd; color: #1565c0; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
      .table-action-bar { background: #fdfdfd; border: 1px solid #e2e8e4; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; }
      .prediction-result-box { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 2px solid #2e7d32; border-radius: 16px; padding: 28px; text-align: center; }
      .slider-label { font-weight: 600; color: #1b3820; font-size: 0.88rem; margin-bottom: 4px; display: block; }
      .disease-hero-card { background: linear-gradient(135deg, #b71c1c 0%, #c62828 50%, #e53935 100%); color: white; border-radius: 18px; padding: 26px 32px; box-shadow: 0 8px 25px rgba(183,28,28,0.2); margin-bottom: 22px; }
      .disease-disclaimer { background: #fff8e1; border-left: 4px solid #f57f17; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; }
      .disease-card { background: #fff; border-radius: 12px; border: 1px solid rgba(0,0,0,0.07); padding: 16px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.04); transition: transform 0.15s ease; }
      .disease-card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
      .disease-severity-high   { border-left: 5px solid #c62828; }
      .disease-severity-medium { border-left: 5px solid #f57f17; }
      .disease-severity-low    { border-left: 5px solid #2e7d32; }
      .disease-result-matched  { background: linear-gradient(135deg, #fff3e0, #fff8e1); border: 2px solid #f57f17; border-radius: 16px; padding: 24px; margin-bottom: 16px; }
      .disease-result-healthy  { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 2px solid #2e7d32; border-radius: 16px; padding: 24px; margin-bottom: 16px; }
    "))
  ),

  # ==========================================================================
  # TAB 1: DASHBOARD
  # ==========================================================================
  nav_panel(
    title = "Dashboard",
    icon  = span("📊"),

    div(class = "dashboard-page",
    div(
      class = "dashboard-hero",
      div(class = "row align-items-center",
        div(class = "col-12",
          div(class = "dashboard-eyebrow mb-2", "SmartHarvest AI  /  Field operations"),
          h1("Operations overview", class = "fw-bold mb-2"),
          p(class = "dashboard-hero-copy mb-0", "Monitor active acreage, sales performance, and projected output from one working view.")
        )
      )
    ),

    div(class = "dashboard-toolbar d-flex flex-wrap align-items-center justify-content-between gap-3 mb-4",
      div(
        div(class = "dashboard-section-heading", "Farm snapshot"),
        div(class = "dashboard-section-note", "Totals update automatically as you manage farm records.")
      ),
      downloadButton("btn_export_summary_csv", "Export farm summary", class = "btn btn-success fw-semibold")
    ),

    layout_columns(
      col_widths = c(4, 4, 4),
      div(class = "dashboard-stat-card",
        div(class = "d-flex justify-content-between align-items-start mb-3",
          div(class = "dashboard-stat-label", "Cultivated land"),
          div(class = "dashboard-stat-icon", style = "background:#e8f5ed;color:#247347;", "▰")
        ),
        div(class = "dashboard-stat-value", textOutput("kpi_area_sum")),
        div(class = "dashboard-stat-note mt-1 mb-3", "Across your registered fields"),
        uiOutput("ui_land_progress_bar")
      ),
      div(class = "dashboard-stat-card",
        div(class = "d-flex justify-content-between align-items-start mb-3",
          div(class = "dashboard-stat-label", "Sales revenue"),
          div(class = "dashboard-stat-icon", style = "background:#e8f4fb;color:#2674a5;", "↗")
        ),
        div(class = "dashboard-stat-value", textOutput("kpi_rev_sum")),
        div(class = "dashboard-stat-note mt-2", textOutput("kpi_rev_subtext"))
      ),
      div(class = "dashboard-stat-card",
        div(class = "d-flex justify-content-between align-items-start mb-3",
          div(class = "dashboard-stat-label", "Estimated net profit"),
          div(class = "dashboard-stat-icon", style = "background:#f0edfb;color:#6953a5;", "✦")
        ),
        div(class = "dashboard-stat-value", textOutput("kpi_net_profit")),
        div(class = "dashboard-stat-note mt-2", textOutput("kpi_profit_margin_text"))
      )
    ),

    div(class = "d-flex align-items-end justify-content-between flex-wrap gap-2 mt-4 mb-3",
      div(
        div(class = "dashboard-section-heading", "Field & finance insights"),
        div(class = "dashboard-section-note", "Review operating costs and compare expected output across crops.")
      )
    ),

    div(class = "dashboard-chart-card mb-3",
      div(class = "dashboard-chart-header",
        div(class = "dashboard-chart-title", "Where your money goes"),
        div(class = "dashboard-chart-note", "Expenses grouped by category")
      ),
      div(class = "dashboard-chart-body", plotlyOutput("plot_expense_breakdown", height = "310px"))
    ),

    div(class = "dashboard-chart-card mb-4",
      div(class = "dashboard-chart-header",
        div(class = "dashboard-chart-title", "Expected production by crop"),
        div(class = "dashboard-chart-note", "Estimate based on cultivated area × expected yield per acre; this is not a guaranteed harvest.")
      ),
      div(class = "dashboard-chart-body", plotlyOutput("plot_production_summary", height = "320px"))
    )
    )
  ),

  # ==========================================================================
  # TAB 2: CROP MANAGEMENT
  # ==========================================================================
  nav_panel(
    title = "Crops",
    icon  = span("🌱"),
    div(
      class = "card-modern mb-3",
      div(class = "card-header bg-success text-white", span("🌱 Crop Master Management & Field Records", class = "fw-bold fs-5")),
      div(
        class = "p-4",
        div(
          class = "table-action-bar d-flex justify-content-between align-items-center flex-wrap gap-2",
          div(class = "d-flex gap-2",
            actionButton("btn_add_crop_modal",    "➕ Add New Crop",       class = "btn btn-success fw-semibold"),
            actionButton("btn_edit_crop_modal",   "✏️ Edit Selected Crop", class = "btn btn-outline-primary fw-semibold"),
            actionButton("btn_delete_crop_modal", "🗑️ Delete Selected Crop",class = "btn btn-outline-danger fw-semibold")
          ),
          div(class = "d-flex gap-2 align-items-center",
            downloadButton("btn_download_crops_csv", "📥 Export Crops CSV", class = "btn btn-outline-secondary btn-sm fw-semibold")
          )
        ),
        div(class = "row g-3 mb-3 p-3 bg-light rounded-3 border",
          div(class = "col-md-4", selectInput("filter_crop_status", "Filter by Status:", choices = c("All","Growing","Sown","Ready for Harvest","Harvested"))),
          div(class = "col-md-8 d-flex align-items-center justify-content-end", uiOutput("ui_selected_crop_banner"))
        ),
        div(class = "table-responsive", DTOutput("table_manage_crops"))
      )
    )
  ),

  # ==========================================================================
  # TAB 3: EXPENSES
  # ==========================================================================
  nav_panel(
    title = "Expenses",
    icon  = span("💰"),
    div(
      class = "card-modern mb-3",
      div(class = "card-header bg-danger text-white", span("💰 Farm Expense Management & Cost Control", class = "fw-bold fs-5")),
      div(
        class = "p-4",
        layout_columns(
          col_widths = c(3, 3, 3, 3),
          div(class = "p-3 bg-light rounded-3 border text-center", div(class = "small text-muted fw-bold", "TOTAL EXPENSES"), div(class = "fs-4 fw-bold text-danger", textOutput("exp_tab_total_cost"))),
          div(class = "p-3 bg-light rounded-3 border text-center", div(class = "small text-muted fw-bold", "TOP COST CENTER"), div(class = "fs-4 fw-bold text-primary", textOutput("exp_tab_top_cat"))),
          div(class = "p-3 bg-light rounded-3 border text-center", div(class = "small text-muted fw-bold", "AVG COST / ACRE"),  div(class = "fs-4 fw-bold text-dark",    textOutput("exp_tab_cost_per_acre"))),
          div(class = "p-3 bg-light rounded-3 border text-center", div(class = "small text-muted fw-bold", "TRANSACTIONS"),    div(class = "fs-4 fw-bold text-success",  textOutput("exp_tab_count")))
        ),
        br(),
        div(
          class = "table-action-bar d-flex justify-content-between align-items-center flex-wrap gap-2",
          div(class = "d-flex gap-2",
            actionButton("btn_add_expense_modal",    "➕ Log New Expense",       class = "btn btn-danger fw-semibold"),
            actionButton("btn_delete_expense_modal", "🗑️ Delete Selected Expense", class = "btn btn-outline-danger fw-semibold")
          ),
          downloadButton("btn_download_expenses_csv", "📥 Export Expenses CSV", class = "btn btn-outline-secondary btn-sm fw-semibold")
        ),
        layout_columns(
          col_widths = c(6, 6),
          div(class = "card-modern", div(class = "card-header", "📅 Monthly Expense Trend"), div(class = "p-3", plotlyOutput("plot_tab_expense_monthly", height = "280px"))),
          div(class = "card-modern", div(class = "card-header", "🥧 Category Breakdown"),     div(class = "p-3", plotlyOutput("plot_tab_expense_donut", height = "280px")))
        ),
        div(class = "table-responsive mt-2", DTOutput("table_manage_expenses"))
      )
    )
  ),

  # ==========================================================================
  # TAB 4: HARVEST & SALES
  # ==========================================================================
  nav_panel(
    title = "Harvest & Sales",
    icon  = span("📦"),
    div(
      class = "card-modern mb-4",
      div(class = "card-header bg-primary text-white", span("🌾 Harvest Production Logging", class = "fw-bold fs-5")),
      div(
        class = "p-4",
        div(
          class = "table-action-bar d-flex justify-content-between align-items-center flex-wrap gap-2",
          div(class = "d-flex gap-2",
            actionButton("btn_add_harvest_modal",    "➕ Record New Harvest", class = "btn btn-primary fw-semibold"),
            actionButton("btn_delete_harvest_modal", "🗑️ Delete Harvest",     class = "btn btn-outline-danger fw-semibold")
          ),
          downloadButton("btn_download_harvest_csv", "📥 Export Harvests", class = "btn btn-outline-secondary btn-sm")
        ),
        div(class = "table-responsive", DTOutput("table_manage_harvest"))
      )
    ),
    div(
      class = "card-modern mb-3",
      div(class = "card-header bg-success text-white", span("💰 Crop Sales & Revenue Realization", class = "fw-bold fs-5")),
      div(
        class = "p-4",
        div(
          class = "table-action-bar d-flex justify-content-between align-items-center flex-wrap gap-2",
          div(class = "d-flex gap-2",
            actionButton("btn_add_sale_modal",    "➕ Record New Sale", class = "btn btn-success fw-semibold"),
            actionButton("btn_delete_sale_modal", "🗑️ Delete Sale",     class = "btn btn-outline-danger fw-semibold")
          ),
          downloadButton("btn_download_sales_csv", "📥 Export Sales CSV", class = "btn btn-outline-secondary btn-sm")
        ),
        div(class = "table-responsive", DTOutput("table_manage_sales"))
      )
    )
  ),

  # ==========================================================================
  # TAB 5: SOIL DATA
  # ==========================================================================
  nav_panel(
    title = "Soil Data",
    icon  = span("🧪"),
    div(
      class = "card-modern mb-3",
      div(class = "card-header bg-secondary text-white", span("🧪 Soil Health & Parcel Nutrients", class = "fw-bold fs-5")),
      div(
        class = "p-4",
        div(class = "soil-advisory-alert d-flex align-items-center gap-3",
          span("⚠️", style = "font-size: 1.8rem;"),
          div(class = "small text-muted", "Educational Advisory Notice: Soil health metrics are for research and record-keeping only. Consult agricultural extension officers for field application.")
        ),
        div(
          class = "table-action-bar d-flex justify-content-between align-items-center flex-wrap gap-2",
          div(class = "d-flex gap-2",
            actionButton("btn_add_soil_modal",    "➕ Log New Soil Test",    class = "btn btn-secondary fw-semibold"),
            actionButton("btn_edit_soil_modal",   "✏️ Edit Soil Record",     class = "btn btn-outline-primary fw-semibold"),
            actionButton("btn_delete_soil_modal", "🗑️ Delete Soil Record",   class = "btn btn-outline-danger fw-semibold")
          ),
          downloadButton("btn_download_soil_csv", "📥 Export Soil Data", class = "btn btn-outline-secondary btn-sm")
        ),
        layout_columns(
          col_widths = c(7, 5),
          div(class = "card-modern", div(class = "card-header", "🧪 N-P-K Concentrations (kg/ha)"),  div(class = "p-3", plotlyOutput("plot_soil_npk_bar",    height = "280px"))),
          div(class = "card-modern", div(class = "card-header", "💧 pH vs Moisture Correlation"),     div(class = "p-3", plotlyOutput("plot_soil_ph_scatter", height = "280px")))
        ),
        div(class = "table-responsive mt-2", DTOutput("table_manage_soil"))
      )
    )
  ),

  # ==========================================================================
  # TAB 6: AI YIELD PREDICTION  (PHASE 8)
  # ==========================================================================
  nav_panel(
    title = "Yield Prediction",
    icon  = span("🤖"),

    # Hero Banner
    div(
      class = "ml-hero-card",
      div(
        class = "row align-items-center g-3",
        div(
          class = "col-lg-8",
          div(class = "d-flex align-items-center gap-2 mb-2",
            span(class = "badge bg-white text-primary px-2 py-1 fw-bold", "🤖 AI / ML Powered"),
            span(class = "badge bg-dark bg-opacity-25 text-white px-2 py-1", "Random Forest Model · R")
          ),
          h2("Crop Yield Prediction System", class = "fw-bold mb-1"),
          p(class = "mb-0 opacity-90", "Enter your field conditions below. The trained machine learning model will predict expected yield per acre in kg.")
        ),
        div(
          class = "col-lg-4",
          div(
            class = "row g-2 text-center",
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("🌳 Trees in Forest", class = "small d-block text-white-50"), strong("300 Trees", class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("📊 Training Records", class = "small d-block text-white-50"), strong(textOutput("ml_n_train"), class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("🎯 R² Accuracy", class = "small d-block text-white-50"),       strong(textOutput("ml_r2"),       class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("📏 RMSE Error", class = "small d-block text-white-50"),        strong(textOutput("ml_rmse"),     class = "fs-6 text-white")))
          )
        )
      )
    ),

    # Educational disclaimer
    div(
      class = "ml-advisory-box d-flex align-items-start gap-3 mb-3",
      span("ℹ️", style = "font-size: 1.5rem; margin-top:2px;"),
      div(
        class = "small",
        strong("Educational Model Disclaimer: "),
        "This yield prediction is generated by a Random Forest regression model trained on simulated agronomic data. ",
        "Results are for educational purposes and should not replace professional agronomist advice. ",
        "Actual field yields depend on many factors not captured here."
      )
    ),

    layout_columns(
      col_widths = c(4, 8),

      # --- LEFT: Input Panel ---
      div(
        class = "card-modern",
        div(class = "card-header", span("🎛️ Field Parameter Inputs", class = "fw-bold")),
        div(
          class = "p-4",
          selectInput("pred_crop", tags$span(class = "slider-label", "🌾 Crop Type"),
            choices = c("Wheat","Rice","Cotton","Maize","Tomato","Soybean","Potato"), selected = "Wheat"),
          br(),
          numericInput("pred_area", tags$span(class = "slider-label", "📐 Field Area (Acres)"), value = 10, min = 1, max = 100, step = 0.5),
          br(),
          numericInput("pred_ph",   tags$span(class = "slider-label", "🧪 Soil pH"),             value = 6.8, min = 4.0, max = 9.0, step = 0.1),
          br(),
          numericInput("pred_n",    tags$span(class = "slider-label", "🟢 Nitrogen (kg/ha)"),    value = 260, min = 50, max = 500, step = 5),
          numericInput("pred_p",    tags$span(class = "slider-label", "🟡 Phosphorus (kg/ha)"),  value = 38,  min = 5,  max = 100, step = 1),
          numericInput("pred_k",    tags$span(class = "slider-label", "🔵 Potassium (kg/ha)"),   value = 210, min = 50, max = 400, step = 5),
          br(),
          numericInput("pred_rain", tags$span(class = "slider-label", "🌧️ Seasonal Rainfall (mm)"), value = 600, min = 100, max = 2000, step = 10),
          numericInput("pred_temp", tags$span(class = "slider-label", "🌡️ Avg Temperature (°C)"),  value = 25.0, min = 10, max = 45, step = 0.5),
          br(),
          actionButton("btn_predict_yield", "🚀 Predict Crop Yield", class = "btn btn-primary fw-bold w-100 py-3 fs-5", style = "letter-spacing: 0.5px;")
        )
      ),

      # --- RIGHT: Results Panel ---
      div(
        # Prediction Result Box
        uiOutput("ui_yield_prediction_result"),

        br(),

        layout_columns(
          col_widths = c(6, 6),
          div(class = "card-modern", div(class = "card-header", span("📊 Predicted vs Actual (Test Set)", class = "fw-bold")), div(class = "p-3", plotlyOutput("plot_pred_vs_actual", height = "300px"))),
          div(class = "card-modern", div(class = "card-header", span("🏆 Feature Importance Ranking",     class = "fw-bold")), div(class = "p-3", plotlyOutput("plot_feat_importance",  height = "300px")))
        ),

        div(class = "card-modern",
          div(class = "card-header", span("📈 Yield Distribution by Crop (Training Data)", class = "fw-bold")),
          div(class = "p-3", plotlyOutput("plot_yield_by_crop", height = "280px"))
        )
      )
    )
  ),

  # ==========================================================================
  # TAB 7: DISEASE DETECTION (PHASE 9)
  # ==========================================================================
  nav_panel(
    title = "Disease Detection",
    icon  = span("🔍"),

    # Hero Banner
    div(
      class = "disease-hero-card",
      div(
        class = "row align-items-center g-3",
        div(
          class = "col-lg-8",
          div(class = "d-flex align-items-center gap-2 mb-2",
            span(class = "badge bg-white text-danger px-2 py-1 fw-bold", "🔬 AI-Assisted Analysis"),
            span(class = "badge bg-dark bg-opacity-25 text-white px-2 py-1", "Educational Demonstration Mode")
          ),
          h2("Crop Leaf Disease Detection", class = "fw-bold mb-1"),
          p(class = "mb-0 opacity-90", "Upload a leaf photo, select the crop type, and receive an educational disease diagnosis with treatment guidelines.")
        ),
        div(
          class = "col-lg-4",
          div(class = "row g-2 text-center",
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("🌿 Crops Covered",  class = "small d-block text-white-50"), strong("7 Major Crops",       class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("🦠 Diseases in DB", class = "small d-block text-white-50"), strong("28 Diseases",          class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("💊 Treatments",     class = "small d-block text-white-50"), strong("Organic + Chemical",   class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("⚠️ Mode",          class = "small d-block text-white-50"), strong("Demo / Educational",   class = "fs-6 text-white")))
          )
        )
      )
    ),

    # Disclaimer
    div(
      class = "disease-disclaimer d-flex align-items-start gap-3 mb-3",
      span("⚠️", style = "font-size: 1.6rem; margin-top: 2px;"),
      div(class = "small",
        strong("Important Disclaimer: "),
        "This is an ", strong("educational demonstration"), " using rule-based symptom matching. ",
        "It does NOT use real AI image recognition. Actual field diagnosis must be confirmed by a ",
        strong("certified plant pathologist or agricultural extension officer"),
        " before applying any treatment."
      )
    ),

    layout_columns(
      col_widths = c(4, 8),

      # ---- LEFT: Upload & Controls ----
      div(
        class = "card-modern",
        div(class = "card-header", span("📤 Upload Leaf Image & Select Symptoms", class = "fw-bold")),
        div(
          class = "p-4",
          fileInput("disease_image_upload",
            label      = tags$span(class = "slider-label", "🌿 Select Leaf Photo (JPG / PNG)"),
            accept     = c("image/jpeg","image/png","image/jpg","image/webp"),
            placeholder = "No file selected",
            buttonLabel = "Browse Photo"
          ),
          uiOutput("ui_leaf_image_preview"),
          hr(),
          selectInput("disease_crop_select",
            label   = tags$span(class = "slider-label", "🌾 Crop Type"),
            choices = c("Wheat","Rice","Cotton","Maize","Tomato","Potato","Soybean"),
            selected = "Wheat"
          ),
          br(),
          div(class = "small text-muted mb-2", "📋 Check symptoms observed on the leaf:"),
          checkboxGroupInput("disease_symptoms", label = NULL,
            choices = c(
              "Yellow / pale spots"        = "yellow_spots",
              "Brown or rust-colored patches" = "brown_rust",
              "White powdery coating"      = "white_powder",
              "Dark / black lesions"       = "black_lesions",
              "Wilting or drooping"        = "wilting",
              "Leaf curling or distortion" = "curling",
              "Water-soaked areas"         = "water_soaked",
              "Orange pustules"            = "orange_pustules"
            )
          ),
          br(),
          actionButton("btn_analyze_disease", "🔬 Analyse Symptoms",
            class = "btn btn-danger fw-bold w-100 py-3 fs-5")
        )
      ),

      # ---- RIGHT: Results + Library ----
      div(
        uiOutput("ui_disease_analysis_result"),
        br(),
        div(
          class = "card-modern",
          div(class = "card-header", span("📚 Common Crop Diseases Reference Library", class = "fw-bold")),
          div(class = "p-4", uiOutput("ui_disease_library"))
        )
      )
    )
  ),

  # ==========================================================================
  # TAB 8: REPORTS & ANALYTICS  (PHASE 10)
  # ==========================================================================
  nav_panel(
    title = "Reports",
    icon  = span("📑"),

    # Hero Banner
    div(
      style = "background: linear-gradient(135deg, #4a148c 0%, #6a1b9a 60%, #9c27b0 100%); color: white; border-radius: 18px; padding: 26px 32px; box-shadow: 0 8px 25px rgba(74,20,140,0.2); margin-bottom: 22px;",
      div(
        class = "row align-items-center g-3",
        div(
          class = "col-lg-8",
          div(class = "d-flex align-items-center gap-2 mb-2",
            span(class = "badge bg-white px-2 py-1 fw-bold", style = "color:#6a1b9a;", "📑 Farm Intelligence Reports"),
            span(class = "badge bg-dark bg-opacity-25 text-white px-2 py-1", "Season: Kharif / Rabi 2026")
          ),
          h2("Comprehensive Farm Performance Reports", class = "fw-bold mb-1"),
          p(class = "mb-0 opacity-90", "Export full-season reports, view financial analytics, crop performance summaries, and download data for all modules.")
        ),
        div(
          class = "col-lg-4",
          div(class = "row g-2 text-center",
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("📦 Modules Covered", class = "small d-block text-white-50"), strong("6 Sections", class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("📥 Export Formats", class = "small d-block text-white-50"), strong("CSV / Print", class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("📊 Charts", class = "small d-block text-white-50"), strong("5 Analytics", class = "fs-6 text-white"))),
            div(class = "col-6", div(class = "p-2 bg-white bg-opacity-10 rounded border border-white border-opacity-25", span("🖨️ Print Ready", class = "small d-block text-white-50"), strong("Yes", class = "fs-6 text-white")))
          )
        )
      )
    ),

    # KPI Summary Row
    layout_columns(
      col_widths = c(3, 3, 3, 3),
      div(class = "kpi-container",
        div(class = "d-flex justify-content-between align-items-start",
          div(div(class = "kpi-title", "Total Revenue"),
              div(class = "kpi-value text-success", textOutput("rpt_total_revenue")),
              div(class = "kpi-subtext text-muted", "Realized crop sales")),
          div(class = "kpi-icon-bubble", style = "background:#e8f5e9; color:#2e7d32;", "💰")
        )
      ),
      div(class = "kpi-container",
        div(class = "d-flex justify-content-between align-items-start",
          div(div(class = "kpi-title", "Total Expenses"),
              div(class = "kpi-value text-danger", textOutput("rpt_total_expenses")),
              div(class = "kpi-subtext text-muted", "All cost categories")),
          div(class = "kpi-icon-bubble", style = "background:#ffebee; color:#c62828;", "💸")
        )
      ),
      div(class = "kpi-container",
        div(class = "d-flex justify-content-between align-items-start",
          div(div(class = "kpi-title", "Net Profit / Loss"),
              div(class = "kpi-value", textOutput("rpt_net_profit")),
              div(class = "kpi-subtext text-muted", textOutput("rpt_margin_text"))),
          div(class = "kpi-icon-bubble", style = "background:#e3f2fd; color:#1565c0;", "📈")
        )
      ),
      div(class = "kpi-container",
        div(class = "d-flex justify-content-between align-items-start",
          div(div(class = "kpi-title", "Total Production"),
              div(class = "kpi-value", textOutput("rpt_total_production")),
              div(class = "kpi-subtext text-muted", "Harvested quantity")),
          div(class = "kpi-icon-bubble", style = "background:#fff8e1; color:#f57f17;", "📦")
        )
      )
    ),

    br(),

    # Analytics Charts Row 1
    layout_columns(
      col_widths = c(6, 6),
      div(class = "card-modern",
        div(class = "card-header", span("📊 Monthly Revenue vs Expenses Trend", class = "fw-bold")),
        div(class = "p-3", plotlyOutput("rpt_plot_monthly_trend", height = "300px"))
      ),
      div(class = "card-modern",
        div(class = "card-header", span("🌾 Crop-wise Revenue Contribution", class = "fw-bold")),
        div(class = "p-3", plotlyOutput("rpt_plot_crop_revenue", height = "300px"))
      )
    ),

    # Analytics Charts Row 2
    layout_columns(
      col_widths = c(5, 7),
      div(class = "card-modern",
        div(class = "card-header", span("💸 Expense Category Share", class = "fw-bold")),
        div(class = "p-3", plotlyOutput("rpt_plot_expense_cat", height = "280px"))
      ),
      div(class = "card-modern",
        div(class = "card-header", span("🏆 Yield Performance: Actual vs Expected (kg/acre)", class = "fw-bold")),
        div(class = "p-3", plotlyOutput("rpt_plot_yield_compare", height = "280px"))
      )
    ),

    div(class = "card-modern",
      div(class = "card-header", span("📋 Profitability by Crop", class = "fw-bold")),
      div(class = "p-3", plotlyOutput("rpt_plot_profit_by_crop", height = "260px"))
    ),

    # Data Export Section
    div(
      class = "card-modern",
      div(class = "card-header", span("📥 Data Export Centre", class = "fw-bold")),
      div(
        class = "p-4",
        div(
          class = "soil-advisory-alert d-flex align-items-center gap-3 mb-4",
          span("ℹ️", style = "font-size: 1.6rem;"),
          div(class = "small text-muted",
            strong("Export Guide: "),
            "Download individual module CSVs or the full consolidated farm report. ",
            "Use the Print Report button to get a browser-printable version of this page."
          )
        ),

        layout_columns(
          col_widths = c(4, 4, 4),
          div(
            class = "p-3 border rounded-3 bg-light text-center",
            div(class = "fs-2 mb-2", "🌾"),
            div(class = "fw-bold mb-1", "Crop Records"),
            div(class = "small text-muted mb-3", "All registered crops with sowing, harvest dates and yield"),
            downloadButton("rpt_dl_crops", "📥 Download Crops CSV", class = "btn btn-outline-success btn-sm w-100 fw-semibold")
          ),
          div(
            class = "p-3 border rounded-3 bg-light text-center",
            div(class = "fs-2 mb-2", "💸"),
            div(class = "fw-bold mb-1", "Expense Ledger"),
            div(class = "small text-muted mb-3", "Full itemised expense log with categories and dates"),
            downloadButton("rpt_dl_expenses", "📥 Download Expenses CSV", class = "btn btn-outline-danger btn-sm w-100 fw-semibold")
          ),
          div(
            class = "p-3 border rounded-3 bg-light text-center",
            div(class = "fs-2 mb-2", "💰"),
            div(class = "fw-bold mb-1", "Sales Invoices"),
            div(class = "small text-muted mb-3", "All crop sales with quantity, unit price and revenue"),
            downloadButton("rpt_dl_sales", "📥 Download Sales CSV", class = "btn btn-outline-primary btn-sm w-100 fw-semibold")
          )
        ),

        br(),

        layout_columns(
          col_widths = c(4, 4, 4),
          div(
            class = "p-3 border rounded-3 bg-light text-center",
            div(class = "fs-2 mb-2", "📦"),
            div(class = "fw-bold mb-1", "Harvest Batches"),
            div(class = "small text-muted mb-3", "Harvest production records with quality grades"),
            downloadButton("rpt_dl_harvest", "📥 Download Harvest CSV", class = "btn btn-outline-warning btn-sm w-100 fw-semibold")
          ),
          div(
            class = "p-3 border rounded-3 bg-light text-center",
            div(class = "fs-2 mb-2", "🧪"),
            div(class = "fw-bold mb-1", "Soil Tests"),
            div(class = "small text-muted mb-3", "N-P-K values, pH, moisture and field health status"),
            downloadButton("rpt_dl_soil", "📥 Download Soil CSV", class = "btn btn-outline-secondary btn-sm w-100 fw-semibold")
          ),
          div(
            class = "p-3 border rounded-3 bg-success text-white text-center",
            div(class = "fs-2 mb-2", "📑"),
            div(class = "fw-bold mb-1", "Full Farm Report"),
            div(class = "small mb-3", "Consolidated all-modules report for the season"),
            downloadButton("rpt_dl_full", "📥 Download Full Report", class = "btn btn-light btn-sm w-100 fw-bold text-success")
          )
        )
      )
    ),

    # Season Summary Table
    div(
      class = "card-modern",
      div(class = "card-header d-flex justify-content-between align-items-center",
        span("📋 Season Crop Performance Summary", class = "fw-bold"),
        actionButton("rpt_btn_print", "🖨️ Print This Report",
          class = "btn btn-outline-secondary btn-sm fw-semibold",
          onclick = "window.print();")
      ),
      div(class = "p-3", DTOutput("rpt_table_summary"))
    )
  )
)

# ------------------------------------------------------------------------------
# 4. SERVER LOGIC
# ------------------------------------------------------------------------------
server <- function(input, output, session) {

  # --- REACTIVE STATE ---
  crops_val    <- reactiveVal(read.csv(crops_csv,    stringsAsFactors = FALSE))
  expenses_val <- reactiveVal(read.csv(expenses_csv, stringsAsFactors = FALSE))
  sales_val    <- reactiveVal(read.csv(sales_csv,    stringsAsFactors = FALSE))
  harvest_val  <- reactiveVal(read.csv(harvest_csv,  stringsAsFactors = FALSE))
  soil_val     <- reactiveVal(read.csv(soil_csv,     stringsAsFactors = FALSE))

  # Persist helpers
  save_crops_to_disk    <- function(df) { crops_val(df);    write.csv(df, crops_csv,    row.names = FALSE) }
  save_expenses_to_disk <- function(df) { expenses_val(df); write.csv(df, expenses_csv, row.names = FALSE) }
  save_harvest_to_disk  <- function(df) { harvest_val(df);  write.csv(df, harvest_csv,  row.names = FALSE) }
  save_sales_to_disk    <- function(df) { sales_val(df);    write.csv(df, sales_csv,    row.names = FALSE) }
  save_soil_to_disk     <- function(df) { soil_val(df);     write.csv(df, soil_csv,     row.names = FALSE) }

  # --- DASHBOARD ---
  output$ui_land_progress_bar <- renderUI({
    df <- crops_val()
    tot_acres <- if (nrow(df) > 0) sum(df$Area_Acres, na.rm = TRUE) else 0
    capacity_pct <- if (tot_acres > 0) round((tot_acres / 80) * 100) else 0
    bar_pct <- max(0, min(100, capacity_pct))
    tagList(
      div(class = "d-flex justify-content-between dashboard-stat-note mb-2",
        span("Land in use"),
        span(paste0(format(round(tot_acres, 1), nsmall = 1), " / 80 acres"))
      ),
      div(class = "progress dashboard-land-progress",
        div(
          class = "progress-bar",
          role = "progressbar",
          `aria-valuenow` = bar_pct,
          `aria-valuemin` = 0,
          `aria-valuemax` = 100,
          style = paste0("width: ", bar_pct, "%;")
        )
      ),
      div(class = "dashboard-stat-note mt-2", paste0(capacity_pct, "% of farm capacity"))
    )
  })

  output$kpi_area_sum <- renderText({
    df <- crops_val()
    area <- if (nrow(df) > 0) sum(df$Area_Acres, na.rm = TRUE) else 0
    paste(format(round(area, 1), nsmall = 1), "acres")
  })
  output$kpi_rev_sum <- renderText({
    df <- sales_val()
    total <- if (nrow(df) > 0) sum(df$Total_Revenue, na.rm = TRUE) else 0
    paste0("₹ ", format(round(total), big.mark = ","))
  })
  output$kpi_rev_subtext <- renderText({
    df <- sales_val()
    quantity <- if (nrow(df) > 0) sum(df$Quantity_Kg, na.rm = TRUE) else 0
    paste0(format(round(quantity), big.mark = ","), " kg sold")
  })
  output$kpi_net_profit        <- renderText({
    rev <- if(nrow(sales_val())>0) sum(sales_val()$Total_Revenue,na.rm=TRUE) else 0
    exp <- if(nrow(expenses_val())>0) sum(expenses_val()$Amount,na.rm=TRUE) else 0
    p <- rev - exp; s <- if(p>=0) "+₹ " else "-₹ "
    paste0(s, format(round(abs(p)), big.mark=","))
  })
  output$kpi_profit_margin_text <- renderText({
    rev <- if(nrow(sales_val())>0) sum(sales_val()$Total_Revenue,na.rm=TRUE) else 0
    exp <- if(nrow(expenses_val())>0) sum(expenses_val()$Amount,na.rm=TRUE) else 0
    expense_text <- paste0("After ₹ ", format(round(exp), big.mark = ","), " expenses")
    if(rev>0) paste0(expense_text, " · ", round(((rev-exp)/rev)*100,1), "% margin") else paste(expense_text, "· no sales recorded")
  })

  output$plot_expense_breakdown <- renderPlotly({
    df <- expenses_val()
    validate(need(nrow(df) > 0, "Add expense records to see spending by category."))
    s <- df %>%
      group_by(Category) %>%
      summarise(Total = sum(Amount, na.rm = TRUE), .groups = "drop") %>%
      arrange(Total)
    s$Category <- factor(s$Category, levels = s$Category)
    plot_ly(
      s,
      x = ~Total,
      y = ~Category,
      type = "bar",
      orientation = "h",
      text = ~paste0("₹ ", format(round(Total), big.mark = ",")),
      textposition = "outside",
      hovertemplate = "%{y}<br>₹ %{x:,.0f}<extra></extra>",
      cliponaxis = FALSE,
      marker = list(color = "#cf815f", line = list(width = 0))
    ) %>%
      layout(
        xaxis = list(title = list(text = ""), range = c(0, max(s$Total) * 1.24), showgrid = TRUE, gridcolor = "#edf1ee", zeroline = FALSE),
        yaxis = list(title = list(text = ""), automargin = TRUE),
        margin = list(l = 12, r = 24, t = 8, b = 24),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = list(color = "#53645a")
      ) %>%
      config(displayModeBar = FALSE, responsive = TRUE)
  })
  output$plot_production_summary <- renderPlotly({
    df <- crops_val()
    validate(need(nrow(df) > 0, "Add crop records to see expected production."))
    d2 <- df %>%
      mutate(Total_Kg = Area_Acres * Yield_Per_Acre_Kg) %>%
      arrange(Total_Kg)
    d2$Crop_Name <- factor(d2$Crop_Name, levels = d2$Crop_Name)
    plot_ly(
      d2,
      x = ~Total_Kg,
      y = ~Crop_Name,
      type = "bar",
      orientation = "h",
      text = ~paste0(format(round(Total_Kg), big.mark = ","), " kg"),
      textposition = "outside",
      hovertemplate = "%{y}<br>%{x:,.0f} kg estimated<extra></extra>",
      cliponaxis = FALSE,
      marker = list(color = "#397b70", line = list(width = 0))
    ) %>%
      layout(
        xaxis = list(title = list(text = ""), range = c(0, max(d2$Total_Kg) * 1.18), showgrid = TRUE, gridcolor = "#edf1ee", zeroline = FALSE),
        yaxis = list(title = list(text = ""), automargin = TRUE),
        margin = list(l = 12, r = 24, t = 8, b = 24),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = list(color = "#53645a")
      ) %>%
      config(displayModeBar = FALSE, responsive = TRUE)
  })

  # --- CROPS SERVER ---
  filtered_manage_crops <- reactive({
    df <- crops_val()
    if (nrow(df)==0) return(df)
    if (input$filter_crop_status != "All") df <- df %>% filter(Status == input$filter_crop_status)
    df
  })

  output$table_manage_crops <- renderDT({
    df <- filtered_manage_crops(); req(nrow(df)>0)
    df$Status_Badge <- paste0('<span class="badge-soft-success">● ', df$Status, '</span>')
    datatable(df %>% select(`Crop ID`=Crop_ID,`Crop Name`=Crop_Name,Variety,`Area (Acres)`=Area_Acres,`Sowing Date`=Sowing_Date,`Expected Harvest`=Expected_Harvest_Date,`Yield/Acre (Kg)`=Yield_Per_Acre_Kg,Status=Status_Badge),
      escape=FALSE, selection="single", options=list(pageLength=8,dom="Bfrtip"), class="table table-hover align-middle", rownames=FALSE)
  })
  output$ui_selected_crop_banner <- renderUI({
    sel <- input$table_manage_crops_rows_selected
    if (length(sel)==0) span(class="badge bg-secondary p-2","Click row to Select, Edit, or Delete")
    else { df <- filtered_manage_crops(); span(class="badge bg-success p-2 fs-6", paste("Selected:",df$Crop_Name[sel],"(",df$Crop_ID[sel],")")) }
  })
  observeEvent(input$btn_add_crop_modal, {
    df <- crops_val()
    next_id <- sprintf("CROP%03d", max(c(0,as.numeric(gsub("CROP","",df$Crop_ID))),na.rm=TRUE)+1)
    showModal(modalDialog(title=div(class="fw-bold text-success","🌱 Register New Crop"),
      div(class="row g-3",
        div(class="col-md-6", textInput("add_crop_id","Crop ID",next_id), textInput("add_crop_name","Crop Name*"), textInput("add_variety","Variety*"), numericInput("add_area","Area (Acres)*",5,0.1)),
        div(class="col-md-6", dateInput("add_sowing_date","Sowing Date*",Sys.Date()), dateInput("add_harvest_date","Expected Harvest*",Sys.Date()+120), selectInput("add_status","Status*",c("Growing","Sown","Ready for Harvest","Harvested")), numericInput("add_yield","Yield/Acre (Kg)*",1800,50))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_add_crop","💾 Save Crop",class="btn btn-success fw-semibold")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_add_crop, {
    new_row <- data.frame(Crop_ID=trimws(input$add_crop_id),Crop_Name=trimws(input$add_crop_name),Variety=trimws(input$add_variety),Area_Acres=as.numeric(input$add_area),Sowing_Date=as.character(input$add_sowing_date),Expected_Harvest_Date=as.character(input$add_harvest_date),Status=input$add_status,Yield_Per_Acre_Kg=as.numeric(input$add_yield),stringsAsFactors=FALSE)
    save_crops_to_disk(rbind(crops_val(),new_row)); removeModal()
    showNotification(paste("Added crop",new_row$Crop_Name),type="message")
  })
  observeEvent(input$btn_edit_crop_modal, {
    sel <- input$table_manage_crops_rows_selected
    if (length(sel)==0) { showNotification("Select a crop first.",type="warning"); return() }
    r <- filtered_manage_crops()[sel,]
    showModal(modalDialog(title=div(class="fw-bold text-primary",paste("✏️ Edit Crop:",r$Crop_Name)),
      div(class="row g-3",
        div(class="col-md-6",textInput("edit_crop_id","Crop ID",r$Crop_ID),textInput("edit_crop_name","Crop Name",r$Crop_Name),textInput("edit_variety","Variety",r$Variety),numericInput("edit_area","Area (Acres)",r$Area_Acres)),
        div(class="col-md-6",dateInput("edit_sowing_date","Sowing Date",as.Date(r$Sowing_Date)),dateInput("edit_harvest_date","Expected Harvest",as.Date(r$Expected_Harvest_Date)),selectInput("edit_status","Status",c("Growing","Sown","Ready for Harvest","Harvested"),selected=r$Status),numericInput("edit_yield","Yield/Acre (Kg)",r$Yield_Per_Acre_Kg))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_edit_crop","💾 Update Crop",class="btn btn-primary fw-semibold")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_edit_crop, {
    df <- crops_val(); idx <- which(df$Crop_ID==input$edit_crop_id); if(length(idx)==0) return()
    df$Crop_Name[idx]<-trimws(input$edit_crop_name); df$Variety[idx]<-trimws(input$edit_variety)
    df$Area_Acres[idx]<-as.numeric(input$edit_area); df$Sowing_Date[idx]<-as.character(input$edit_sowing_date)
    df$Expected_Harvest_Date[idx]<-as.character(input$edit_harvest_date); df$Status[idx]<-input$edit_status
    df$Yield_Per_Acre_Kg[idx]<-as.numeric(input$edit_yield)
    save_crops_to_disk(df); removeModal(); showNotification("Crop updated!",type="message")
  })
  observeEvent(input$btn_delete_crop_modal, {
    sel <- input$table_manage_crops_rows_selected
    if (length(sel)==0) { showNotification("Select a crop first.",type="warning"); return() }
    r <- filtered_manage_crops()[sel,]
    showModal(modalDialog(title=div(class="fw-bold text-danger","🗑️ Confirm Delete"),
      p(paste("Delete crop",r$Crop_Name,"(",r$Crop_ID,") permanently?")),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_delete_crop","Delete",class="btn btn-danger")),size="s",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_delete_crop, {
    sel <- input$table_manage_crops_rows_selected; df <- filtered_manage_crops()
    save_crops_to_disk(crops_val() %>% filter(Crop_ID != df$Crop_ID[sel]))
    removeModal(); showNotification("Crop deleted.",type="warning")
  })
  output$btn_download_crops_csv <- downloadHandler(
    filename=function() paste0("crop_records_",Sys.Date(),".csv"),
    content =function(file) write.csv(crops_val(),file,row.names=FALSE)
  )

  # --- EXPENSES SERVER ---
  output$exp_tab_total_cost   <- renderText({ df<-expenses_val(); if(nrow(df)>0) paste0("₹ ",format(round(sum(df$Amount,na.rm=TRUE)),big.mark=",")) else "₹ 0" })
  output$exp_tab_top_cat      <- renderText({ df<-expenses_val(); if(nrow(df)>0){top<-df%>%group_by(Category)%>%summarise(Tot=sum(Amount,na.rm=TRUE))%>%arrange(desc(Tot))%>%slice(1);paste0(top$Category," (₹ ",format(round(top$Tot),big.mark=","),")")}else"None"})
  output$exp_tab_cost_per_acre<- renderText({ te<-sum(expenses_val()$Amount,na.rm=TRUE);ta<-sum(crops_val()$Area_Acres,na.rm=TRUE);if(ta>0)paste0("₹ ",format(round(te/ta),big.mark=","),"/Acre")else"₹ 0"})
  output$exp_tab_count        <- renderText({ paste(nrow(expenses_val()),"Entries") })
  output$plot_tab_expense_monthly <- renderPlotly({
    df<-expenses_val();req(nrow(df)>0)
    m<-df%>%mutate(Month=substr(Date,1,7))%>%group_by(Month)%>%summarise(Tot=sum(Amount,na.rm=TRUE))
    plot_ly(m,x=~Month,y=~Tot,type="bar",marker=list(color="#e57373"))%>%layout(xaxis=list(title="Month"),yaxis=list(title="Amount (₹)"))
  })
  output$plot_tab_expense_donut <- renderPlotly({
    df<-expenses_val();req(nrow(df)>0)
    c<-df%>%group_by(Category)%>%summarise(Tot=sum(Amount,na.rm=TRUE))
    plot_ly(c,labels=~Category,values=~Tot,type="pie",hole=0.5)
  })
  output$table_manage_expenses <- renderDT({
    df<-expenses_val();req(nrow(df)>0)
    df$Formatted_Amount<-paste0("₹ ",format(round(df$Amount),big.mark=","))
    datatable(df%>%select(`Expense ID`=Expense_ID,Category,Description,`Amount`=Formatted_Amount,Date),
      selection="single",options=list(pageLength=8,dom="Bfrtip"),class="table table-hover align-middle",rownames=FALSE)
  })
  observeEvent(input$btn_add_expense_modal, {
    df<-expenses_val()
    next_id<-sprintf("EXP%03d",max(c(0,as.numeric(gsub("EXP","",df$Expense_ID))),na.rm=TRUE)+1)
    showModal(modalDialog(title=div(class="fw-bold text-danger","💸 Log Farm Expense"),
      div(class="row g-3",
        div(class="col-md-6",textInput("add_exp_id","Expense ID",next_id),selectInput("add_exp_cat","Category*",c("Seeds","Fertilizers","Labor","Irrigation","Pesticides","Equipment","Other")),textInput("add_exp_desc","Description*")),
        div(class="col-md-6",numericInput("add_exp_amt","Amount (₹)*",5000,10),dateInput("add_exp_date","Date*",Sys.Date()))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_add_exp","💾 Save Expense",class="btn btn-danger")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_add_exp, {
    new_row<-data.frame(Expense_ID=trimws(input$add_exp_id),Category=input$add_exp_cat,Description=trimws(input$add_exp_desc),Amount=as.numeric(input$add_exp_amt),Date=as.character(input$add_exp_date),Crop_ID="All",stringsAsFactors=FALSE)
    save_expenses_to_disk(rbind(expenses_val(),new_row)); removeModal()
    showNotification("Expense logged!",type="message")
  })
  observeEvent(input$btn_delete_expense_modal, {
    sel<-input$table_manage_expenses_rows_selected
    if(length(sel)==0) return()
    save_expenses_to_disk(expenses_val()[-sel,]); showNotification("Expense deleted.",type="warning")
  })
  output$btn_download_expenses_csv <- downloadHandler(
    filename=function() paste0("expense_records_",Sys.Date(),".csv"),
    content =function(file) write.csv(expenses_val(),file,row.names=FALSE)
  )

  # --- HARVEST & SALES SERVER ---
  output$table_manage_harvest <- renderDT({
    df<-harvest_val();req(nrow(df)>0)
    datatable(df,selection="single",options=list(pageLength=5,dom="Bfrtip"),rownames=FALSE)
  })
  output$table_manage_sales <- renderDT({
    df<-sales_val();req(nrow(df)>0)
    df$Rev<-paste0("₹ ",format(round(df$Total_Revenue),big.mark=","))
    datatable(df%>%select(`Sale ID`=Sale_ID,`Crop Name`=Crop_Name,Date=Sale_Date,`Quantity (Kg)`=Quantity_Kg,`Unit Price (₹)`=Unit_Price,`Total Revenue`=Rev),
      selection="single",options=list(pageLength=5,dom="Bfrtip"),rownames=FALSE)
  })
  observeEvent(input$btn_add_harvest_modal, {
    df<-harvest_val()
    next_id<-sprintf("HARV%03d",max(c(0,as.numeric(gsub("HARV","",df$Harvest_ID))),na.rm=TRUE)+1)
    showModal(modalDialog(title=div(class="fw-bold text-primary","🌾 Record Harvest"),
      div(class="row g-3",
        div(class="col-md-6",textInput("add_h_id","Harvest ID",next_id),textInput("add_h_crop","Crop Name","Wheat"),dateInput("add_h_date","Date",Sys.Date())),
        div(class="col-md-6",numericInput("add_h_qty","Quantity (Kg)*",10000,100),numericInput("add_h_yield","Yield/Acre (Kg)*",2000,100),selectInput("add_h_grade","Grade",c("Grade A+","Grade A","Grade B","Grade C")))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_add_h","Save Harvest",class="btn btn-primary")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_add_h, {
    new_row<-data.frame(Harvest_ID=input$add_h_id,Crop_ID="CROP001",Crop_Name=input$add_h_crop,Harvest_Date=as.character(input$add_h_date),Quantity_Kg=as.numeric(input$add_h_qty),Yield_Per_Acre_Kg=as.numeric(input$add_h_yield),Quality_Grade=input$add_h_grade,stringsAsFactors=FALSE)
    save_harvest_to_disk(rbind(harvest_val(),new_row)); removeModal()
    showNotification("Harvest batch saved!",type="message")
  })
  observeEvent(input$btn_delete_harvest_modal, {
    sel<-input$table_manage_harvest_rows_selected
    if(length(sel)==0) return()
    save_harvest_to_disk(harvest_val()[-sel,]); showNotification("Harvest deleted.",type="warning")
  })
  observeEvent(input$btn_add_sale_modal, {
    df<-sales_val()
    next_id<-sprintf("SALE%03d",max(c(0,as.numeric(gsub("SALE","",df$Sale_ID))),na.rm=TRUE)+1)
    showModal(modalDialog(title=div(class="fw-bold text-success","💰 Record Sale"),
      div(class="row g-3",
        div(class="col-md-6",textInput("add_s_id","Sale ID",next_id),textInput("add_s_crop","Crop Name","Wheat"),dateInput("add_s_date","Date",Sys.Date())),
        div(class="col-md-6",numericInput("add_s_qty","Quantity (Kg)*",5000,100),numericInput("add_s_price","Price / Kg (₹)*",28,1))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_add_s","Save Invoice",class="btn btn-success")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_add_s, {
    qty<-as.numeric(input$add_s_qty);prc<-as.numeric(input$add_s_price)
    new_row<-data.frame(Sale_ID=input$add_s_id,Crop_ID="CROP001",Crop_Name=input$add_s_crop,Sale_Date=as.character(input$add_s_date),Quantity_Kg=qty,Unit_Price=prc,Total_Revenue=qty*prc,stringsAsFactors=FALSE)
    save_sales_to_disk(rbind(sales_val(),new_row)); removeModal()
    showNotification("Sale invoice saved!",type="message")
  })
  observeEvent(input$btn_delete_sale_modal, {
    sel<-input$table_manage_sales_rows_selected
    if(length(sel)==0) return()
    save_sales_to_disk(sales_val()[-sel,]); showNotification("Sale deleted.",type="warning")
  })
  output$btn_download_harvest_csv <- downloadHandler(filename=function() paste0("harvest_records_",Sys.Date(),".csv"), content=function(file) write.csv(harvest_val(),file,row.names=FALSE))
  output$btn_download_sales_csv   <- downloadHandler(filename=function() paste0("sales_records_",Sys.Date(),".csv"),   content=function(file) write.csv(sales_val(),file,row.names=FALSE))

  # --- SOIL SERVER ---
  output$plot_soil_npk_bar <- renderPlotly({
    df<-soil_val();req(nrow(df)>0)
    plot_ly(df,x=~Field_Name) %>%
      add_bars(y=~Nitrogen_kg_ha,   name="N (Nitrogen)",   marker=list(color="#2e7d32")) %>%
      add_bars(y=~Phosphorus_kg_ha, name="P (Phosphorus)", marker=list(color="#f57f17")) %>%
      add_bars(y=~Potassium_kg_ha,  name="K (Potassium)",  marker=list(color="#1565c0")) %>%
      layout(barmode="group",xaxis=list(title="",tickangle=-15),yaxis=list(title="kg/ha"))
  })
  output$plot_soil_ph_scatter <- renderPlotly({
    df<-soil_val();req(nrow(df)>0)
    plot_ly(df,x=~pH,y=~Moisture_Percent,text=~Field_Name,type="scatter",mode="markers+text",marker=list(size=14,color="#2e7d32")) %>%
      layout(xaxis=list(title="Soil pH"),yaxis=list(title="Moisture (%)"))
  })
  output$table_manage_soil <- renderDT({
    df<-soil_val();req(nrow(df)>0)
    df$Status_Badge<-paste0('<span class="badge-soft-success">● ',df$Health_Status,'</span>')
    datatable(df%>%select(`Soil ID`=Soil_ID,Field=Field_Name,`Soil Type`=Soil_Type,pH,N=Nitrogen_kg_ha,P=Phosphorus_kg_ha,K=Potassium_kg_ha,`Moisture (%)`=Moisture_Percent,Status=Status_Badge),
      escape=FALSE,selection="single",options=list(pageLength=5,dom="Bfrtip"),rownames=FALSE)
  })
  observeEvent(input$btn_add_soil_modal, {
    df<-soil_val()
    next_id<-sprintf("SOIL%03d",max(c(0,as.numeric(gsub("SOIL","",df$Soil_ID))),na.rm=TRUE)+1)
    showModal(modalDialog(title=div(class="fw-bold text-secondary","🧪 Log Soil Test"),
      div(class="row g-3",
        div(class="col-md-6",textInput("add_soil_id","Soil ID",next_id),textInput("add_soil_field","Field Name*","East Plot"),selectInput("add_soil_type","Soil Type",c("Clay Loam","Sandy Loam","Alluvial","Black Cotton","Red Sandy Loam"))),
        div(class="col-md-6",numericInput("add_soil_ph","Soil pH*",6.8,1,14,0.1),numericInput("add_soil_n","Nitrogen (N)*",250),numericInput("add_soil_p","Phosphorus (P)*",30),numericInput("add_soil_k","Potassium (K)*",200),numericInput("add_soil_moist","Moisture %*",25))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_add_soil","Save Record",class="btn btn-secondary")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_add_soil, {
    ph_v<-as.numeric(input$add_soil_ph)
    st<-if(ph_v>=6.5&&ph_v<=7.5)"Optimal"else if(ph_v>7.5)"Alkaline"else"Slightly Acidic"
    new_row<-data.frame(Soil_ID=input$add_soil_id,Field_Name=input$add_soil_field,Linked_Crop_ID="CROP001",Soil_Type=input$add_soil_type,pH=ph_v,Nitrogen_kg_ha=as.numeric(input$add_soil_n),Phosphorus_kg_ha=as.numeric(input$add_soil_p),Potassium_kg_ha=as.numeric(input$add_soil_k),Moisture_Percent=as.numeric(input$add_soil_moist),Test_Date=as.character(Sys.Date()),Health_Status=st,stringsAsFactors=FALSE)
    save_soil_to_disk(rbind(soil_val(),new_row)); removeModal()
    showNotification("Soil record saved!",type="message")
  })
  observeEvent(input$btn_edit_soil_modal, {
    sel <- input$table_manage_soil_rows_selected
    if (length(sel)==0) { showNotification("Select a record first.", type="warning"); return() }
    r <- soil_val()[sel,]
    showModal(modalDialog(title=div(class="fw-bold text-primary","✏️ Edit Soil Record"),
      div(class="row g-3",
        div(class="col-md-6",textInput("edit_soil_id","Soil ID",r$Soil_ID),textInput("edit_soil_field","Field Name",r$Field_Name),selectInput("edit_soil_type","Soil Type",c("Clay Loam","Sandy Loam","Alluvial","Black Cotton","Red Sandy Loam"),selected=r$Soil_Type)),
        div(class="col-md-6",numericInput("edit_soil_ph","Soil pH",r$pH,1,14,0.1),numericInput("edit_soil_n","Nitrogen",r$Nitrogen_kg_ha),numericInput("edit_soil_p","Phosphorus",r$Phosphorus_kg_ha),numericInput("edit_soil_k","Potassium",r$Potassium_kg_ha),numericInput("edit_soil_moist","Moisture %",r$Moisture_Percent))
      ),
      footer=tagList(modalButton("Cancel"),actionButton("btn_confirm_edit_soil","💾 Update Record",class="btn btn-primary")),size="m",easyClose=TRUE))
  })
  observeEvent(input$btn_confirm_edit_soil, {
    df<-soil_val(); idx<-which(df$Soil_ID==input$edit_soil_id); if(length(idx)==0) return()
    ph_v<-as.numeric(input$edit_soil_ph)
    df$Field_Name[idx]<-input$edit_soil_field; df$Soil_Type[idx]<-input$edit_soil_type
    df$pH[idx]<-ph_v; df$Nitrogen_kg_ha[idx]<-as.numeric(input$edit_soil_n)
    df$Phosphorus_kg_ha[idx]<-as.numeric(input$edit_soil_p); df$Potassium_kg_ha[idx]<-as.numeric(input$edit_soil_k)
    df$Moisture_Percent[idx]<-as.numeric(input$edit_soil_moist)
    df$Health_Status[idx]<-if(ph_v>=6.5&&ph_v<=7.5)"Optimal"else if(ph_v>7.5)"Alkaline"else"Slightly Acidic"
    save_soil_to_disk(df); removeModal(); showNotification("Soil record updated!",type="message")
  })
  observeEvent(input$btn_delete_soil_modal, {
    sel<-input$table_manage_soil_rows_selected
    if(length(sel)==0){showNotification("Select a record first.",type="warning");return()}
    save_soil_to_disk(soil_val()[-sel,]); showNotification("Soil record deleted.",type="warning")
  })
  output$btn_download_soil_csv <- downloadHandler(filename=function() paste0("soil_records_",Sys.Date(),".csv"), content=function(file) write.csv(soil_val(),file,row.names=FALSE))

  # --- ML YIELD PREDICTION SERVER (PHASE 8) ---
  output$ml_n_train <- renderText({ paste0(ml_result$n_train, " Records") })
  output$ml_r2      <- renderText({ paste0(ml_result$r2 * 100, "% R²") })
  output$ml_rmse    <- renderText({ paste0(ml_result$rmse, " kg/acre") })

  # Prediction result (triggered by button click)
  prediction_rv <- reactiveVal(NULL)

  observeEvent(input$btn_predict_yield, {
    req(input$pred_crop, input$pred_area, input$pred_ph,
        input$pred_n, input$pred_p, input$pred_k,
        input$pred_rain, input$pred_temp)

    new_data <- data.frame(
      Crop             = factor(input$pred_crop, levels = c("Wheat","Rice","Cotton","Maize","Tomato","Soybean","Potato")),
      Area_Acres       = as.numeric(input$pred_area),
      Soil_pH          = as.numeric(input$pred_ph),
      Nitrogen_kg_ha   = as.numeric(input$pred_n),
      Phosphorus_kg_ha = as.numeric(input$pred_p),
      Potassium_kg_ha  = as.numeric(input$pred_k),
      Rainfall_mm      = as.numeric(input$pred_rain),
      Temperature_C    = as.numeric(input$pred_temp),
      stringsAsFactors = FALSE
    )

    pred_yield <- round(predict(ml_result$model, newdata = new_data))
    total_prod <- round(pred_yield * as.numeric(input$pred_area))
    prediction_rv(list(yield = pred_yield, total = total_prod, crop = input$pred_crop, area = input$pred_area))
  })

  output$ui_yield_prediction_result <- renderUI({
    pred <- prediction_rv()
    if (is.null(pred)) {
      div(
        class = "card-modern",
        div(class = "card-header", span("🤖 Prediction Result", class = "fw-bold")),
        div(class = "p-5 text-center text-muted",
          span("🚀", style = "font-size: 3rem; display: block; margin-bottom: 10px;"),
          h5("Enter your field parameters and click"),
          strong("Predict Crop Yield", class = "text-primary fs-5"),
          p(class = "mt-2 small", "The Random Forest model will analyse your inputs and generate an accurate yield estimate.")
        )
      )
    } else {
      # Colour feedback based on yield level
      df_ref <- read.csv(yield_csv, stringsAsFactors = FALSE)
      crop_df <- df_ref[df_ref$Crop == pred$crop, ]
      avg_y   <- mean(crop_df$Actual_Yield_kg_per_acre, na.rm = TRUE)
      pct_diff<- round((pred$yield - avg_y) / avg_y * 100, 1)
      color   <- if (pct_diff >= 5) "#2e7d32" else if (pct_diff >= -5) "#f57f17" else "#c62828"
      status  <- if (pct_diff >= 5) "Above Average Yield 🎉" else if (pct_diff >= -5) "Average Yield 📊" else "Below Average — Review Inputs ⚠️"

      div(
        class = "prediction-result-box mb-4",
        div(class = "d-flex justify-content-center align-items-center gap-3 flex-wrap",
          div(
            div(class = "small fw-bold text-muted", "PREDICTED YIELD PER ACRE"),
            div(style = paste0("font-size: 3.5rem; font-weight: 900; color:", color, "; line-height: 1;"), paste0(format(pred$yield, big.mark=","), " kg")),
            div(class = "small text-muted mt-1", paste("per acre ·", pred$crop))
          ),
          div(style = "width: 2px; height: 80px; background: #ccc; display: none;"),
          div(
            div(class = "small fw-bold text-muted", "ESTIMATED TOTAL FIELD PRODUCTION"),
            div(style = "font-size: 2.2rem; font-weight: 800; color: #1565c0;", paste0(format(pred$total, big.mark=","), " kg")),
            div(class = "small text-muted mt-1", paste("for", pred$area, "Acre Field"))
          )
        ),
        div(class = "mt-3", span(class = "badge px-3 py-2 fw-semibold fs-6", style = paste0("background:", color, "; color: white; border-radius: 20px;"), status)),
        div(class = "small text-muted mt-2", paste0(
          "Crop average yield: ", round(avg_y), " kg/acre · Your prediction: ",
          if(pct_diff >= 0) paste0("+", pct_diff, "%") else paste0(pct_diff, "%"), " vs average"
        ))
      )
    }
  })

  # Predicted vs Actual scatter plot
  output$plot_pred_vs_actual <- renderPlotly({
    pa_df <- data.frame(
      Actual    = ml_result$test_actual,
      Predicted = ml_result$test_preds,
      Crop      = ml_result$test_crop
    )
    min_val <- min(c(pa_df$Actual, pa_df$Predicted))
    max_val <- max(c(pa_df$Actual, pa_df$Predicted))

    plot_ly(pa_df, x = ~Actual, y = ~Predicted, color = ~Crop,
            type = "scatter", mode = "markers",
            marker = list(size = 7, opacity = 0.7)) %>%
      add_trace(x = c(min_val, max_val), y = c(min_val, max_val),
                type = "scatter", mode = "lines",
                name = "Perfect Prediction", line = list(color = "#c62828", dash = "dash", width = 2),
                inherit = FALSE, showlegend = TRUE) %>%
      layout(
        xaxis = list(title = "Actual Yield (kg/acre)"),
        yaxis = list(title = "Predicted Yield (kg/acre)"),
        legend = list(orientation = "h", x = 0, y = -0.25, font = list(size = 10)),
        margin = list(l = 55, r = 20, t = 20, b = 60),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # Feature importance chart
  output$plot_feat_importance <- renderPlotly({
    imp <- importance(ml_result$model)
    imp_df <- data.frame(
      Feature    = rownames(imp),
      Importance = as.numeric(imp[, "%IncMSE"])
    ) %>% arrange(Importance)

    imp_df$Color <- ifelse(imp_df$Importance > median(imp_df$Importance), "#1b5e20", "#81c784")

    plot_ly(imp_df, x = ~Importance, y = ~reorder(Feature, Importance),
            type = "bar", orientation = "h",
            marker = list(color = imp_df$Color)) %>%
      layout(
        xaxis = list(title = "% Increase in MSE (Importance)"),
        yaxis = list(title = ""),
        margin = list(l = 120, r = 20, t = 20, b = 40),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # Yield distribution boxplot by crop
  output$plot_yield_by_crop <- renderPlotly({
    df <- read.csv(yield_csv, stringsAsFactors = FALSE)
    colors <- c("#1b5e20","#1565c0","#f57f17","#c62828","#7b1fa2","#00838f","#4e342e")
    crops  <- unique(df$Crop)

    p <- plot_ly(type = "box")
    for (i in seq_along(crops)) {
      sub <- df[df$Crop == crops[i], ]
      p <- p %>% add_trace(y = sub$Actual_Yield_kg_per_acre, name = crops[i],
                           marker = list(color = colors[i %% length(colors) + 1]))
    }
    p %>% layout(
      yaxis  = list(title = "Actual Yield (kg/acre)"),
      xaxis  = list(title = ""),
      showlegend = FALSE,
      margin = list(l = 55, r = 20, t = 20, b = 40),
      paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
    )
  })

  # --- GLOBAL ACTIONS ---
  output$btn_export_summary_csv <- downloadHandler(
    filename = function() paste0("farm_summary_", Sys.Date(), ".csv"),
    content  = function(file) {
      c_df <- crops_val()
      e_df <- expenses_val()
      s_df <- sales_val()
      h_df <- harvest_val()
      total_revenue <- if (nrow(s_df) > 0) sum(s_df$Total_Revenue, na.rm = TRUE) else 0
      total_expenses <- if (nrow(e_df) > 0) sum(e_df$Amount, na.rm = TRUE) else 0
      total_area <- if (nrow(c_df) > 0) sum(c_df$Area_Acres, na.rm = TRUE) else 0
      total_harvest <- if (nrow(h_df) > 0) sum(h_df$Quantity_Kg, na.rm = TRUE) else 0

      summary <- data.frame(
        Metric = c(
          "Report date",
          "Crops managed",
          "Cultivated land (acres)",
          "Sales revenue (INR)",
          "Farm expenses (INR)",
          "Estimated net profit (INR)",
          "Harvest recorded (kg)"
        ),
        Value = c(
          as.character(Sys.Date()),
          as.character(nrow(c_df)),
          as.character(round(total_area, 1)),
          as.character(round(total_revenue, 2)),
          as.character(round(total_expenses, 2)),
          as.character(round(total_revenue - total_expenses, 2)),
          as.character(round(total_harvest, 2))
        ),
        stringsAsFactors = FALSE
      )
      write.csv(summary, file, row.names = FALSE)
    }
  )

  # ============================================================================
  # PHASE 9: CROP DISEASE DETECTION SERVER LOGIC
  # ============================================================================

  # ---- Disease Knowledge Base ----
  disease_kb <- list(
    Wheat = list(
      list(name="Wheat Rust (Puccinia spp.)", symptoms=c("orange_pustules","brown_rust"), severity="High",
           emoji="🟠", cause="Fungal pathogen spread by wind-borne spores in humid conditions.",
           treatment_organic="Remove and burn infected plant material. Apply neem oil extract spray.",
           treatment_chemical="Apply Propiconazole 25 EC or Tebuconazole 250 EW at 0.1% concentration.",
           prevention="Use rust-resistant varieties (HD-2781, PBW-343). Avoid overhead irrigation.",
           yield_loss="20–70% if untreated"),
      list(name="Powdery Mildew (Blumeria graminis)", symptoms=c("white_powder","yellow_spots"), severity="Medium",
           emoji="⬜", cause="Fungal disease thriving in cool, dry weather with high humidity.",
           treatment_organic="Spray diluted milk solution (1:10) or potassium bicarbonate.",
           treatment_chemical="Apply Sulfur 80 WP or Karathane at recommended dosages.",
           prevention="Ensure good field ventilation. Avoid excess nitrogen fertilization.",
           yield_loss="5–40%"),
      list(name="Loose Smut (Ustilago tritici)", symptoms=c("black_lesions","brown_rust"), severity="Medium",
           emoji="⚫", cause="Seed-borne fungus that destroys grain at heading stage.",
           treatment_organic="Hot water seed treatment at 52°C for 10 minutes.",
           treatment_chemical="Seed treatment with Carboxin + Thiram (Vitavax Power).",
           prevention="Use certified disease-free seeds. Treat seeds before every planting.",
           yield_loss="10–30%"),
      list(name="Septoria Leaf Blotch", symptoms=c("brown_rust","water_soaked","yellow_spots"), severity="Medium",
           emoji="🟤", cause="Fungal pathogen Septoria tritici; spread via rain-splash.",
           treatment_organic="Copper-based fungicides (Bordeaux mixture).",
           treatment_chemical="Propiconazole or Epoxiconazole foliar spray at flag leaf stage.",
           prevention="Rotate with non-cereal crops. Plow under crop residues.",
           yield_loss="15–50%")
    ),
    Rice = list(
      list(name="Rice Blast (Magnaporthe oryzae)", symptoms=c("brown_rust","black_lesions","water_soaked"), severity="High",
           emoji="💥", cause="Most destructive rice fungal disease, favored by warm humid weather.",
           treatment_organic="Silicon soil amendments improve plant resistance.",
           treatment_chemical="Tricyclazole 75 WP or Isoprothiolane 40 EC spray.",
           prevention="Use blast-resistant varieties. Balanced NPK — avoid excess nitrogen.",
           yield_loss="Up to 100% in severe epidemics"),
      list(name="Brown Plant Hopper (BPH) Damage", symptoms=c("wilting","yellow_spots","brown_rust"), severity="High",
           emoji="🟫", cause="Sap-sucking insect pest causing 'hopper burn'; worst in waterlogged fields.",
           treatment_organic="Release natural predators (Cyrtorhinus lividipennis). Light traps at night.",
           treatment_chemical="Buprofezin 25 SC or Thiamethoxam 25 WG.",
           prevention="Avoid excess nitrogen. Drain fields periodically. Maintain field bunds.",
           yield_loss="30–80%"),
      list(name="Sheath Blight (Rhizoctonia solani)", symptoms=c("water_soaked","brown_rust","wilting"), severity="Medium",
           emoji="🌾", cause="Soilborne fungus; spreads via mycelial strands and sclerotia in standing water.",
           treatment_organic="Pseudomonas fluorescens biocontrol spray.",
           treatment_chemical="Hexaconazole 5 EC or Validamycin 3 L.",
           prevention="Reduce plant density. Drain excess water from fields.",
           yield_loss="10–40%"),
      list(name="Bacterial Leaf Blight (Xanthomonas oryzae)", symptoms=c("water_soaked","wilting","yellow_spots"), severity="High",
           emoji="🔴", cause="Bacterial infection spreading through irrigation water and rain splashing.",
           treatment_organic="Copper oxychloride 50 WP spray.",
           treatment_chemical="Streptocycline + Copper oxychloride combination.",
           prevention="Drain flooded areas. Avoid high-density planting. Use resistant varieties.",
           yield_loss="20–50%")
    ),
    Tomato = list(
      list(name="Early Blight (Alternaria solani)", symptoms=c("brown_rust","black_lesions","yellow_spots"), severity="Medium",
           emoji="🍅", cause="Fungal pathogen; brown target-like rings on lower leaves first.",
           treatment_organic="Neem oil (0.5%) foliar spray every 7 days.",
           treatment_chemical="Mancozeb 75 WP or Chlorothalonil 75 WP.",
           prevention="Mulch soil to prevent splash. Remove infected leaves early.",
           yield_loss="20–80%"),
      list(name="Late Blight (Phytophthora infestans)", symptoms=c("water_soaked","wilting","brown_rust"), severity="High",
           emoji="🌿", cause="Oomycete pathogen; rapid spread in cool, wet weather (15–20°C).",
           treatment_organic="Copper-based Bordeaux mixture spray.",
           treatment_chemical="Cymoxanil 8% + Mancozeb 64% (Curzate M-8).",
           prevention="Avoid overhead irrigation. Space plants for airflow. Use resistant hybrids.",
           yield_loss="Up to 100%"),
      list(name="Tomato Mosaic Virus (ToMV)", symptoms=c("yellow_spots","curling"), severity="Medium",
           emoji="🟡", cause="Virus transmitted by contact and aphid vectors.",
           treatment_organic="Remove infected plants immediately. Spray insecticidal soap for aphids.",
           treatment_chemical="No direct cure; control aphid vectors with Imidacloprid 17.8 SL.",
           prevention="Use virus-free certified seeds. Disinfect tools between plants.",
           yield_loss="10–30%"),
      list(name="Fusarium Wilt", symptoms=c("wilting","brown_rust","yellow_spots"), severity="High",
           emoji="🍂", cause="Soilborne fungal pathogen persisting in soil for years.",
           treatment_organic="Trichoderma harzianum soil drench.",
           treatment_chemical="Carbendazim 50 WP soil drench + Propiconazole spray.",
           prevention="Crop rotation (avoid Solanaceae family for 3+ years). Soil solarization.",
           yield_loss="20–60%")
    ),
    Cotton = list(
      list(name="Cotton Leaf Curl Disease (CLCuD)", symptoms=c("curling","yellow_spots"), severity="High",
           emoji="🌀", cause="Begomovirus transmitted by whitefly Bemisia tabaci.",
           treatment_organic="Sticky yellow traps for whitefly monitoring.",
           treatment_chemical="Thiamethoxam 70 WS seed treatment + foliar Spiromesifen.",
           prevention="Use CLCuD-resistant varieties. Rogue out infected plants early.",
           yield_loss="Up to 100% in severely infected fields"),
      list(name="Fusarium Wilt (Fusarium oxysporum f.sp. vasinfectum)", symptoms=c("wilting","yellow_spots"), severity="High",
           emoji="🟤", cause="Soilborne fungal wilt; vascular tissue browning visible on stem cross-section.",
           treatment_organic="Trichoderma viride 2.5 kg/acre soil application.",
           treatment_chemical="Carbendazim 50 WP 2g/L drench around root zone.",
           prevention="Grow resistant hybrids. Maintain soil pH 6.5–7.0. Avoid waterlogging.",
           yield_loss="15–50%"),
      list(name="Alternaria Leaf Spot", symptoms=c("brown_rust","black_lesions","yellow_spots"), severity="Medium",
           emoji="🟠", cause="Fungal disease favored by warm temperatures and high humidity.",
           treatment_organic="Neem oil 2 mL/L spray.",
           treatment_chemical="Mancozeb 75 WP at 2.5g/L.",
           prevention="Remove and destroy infected plant debris. Avoid excess irrigation.",
           yield_loss="10–25%")
    ),
    Maize = list(
      list(name="Northern Corn Leaf Blight (NCLB)", symptoms=c("brown_rust","yellow_spots"), severity="Medium",
           emoji="🌽", cause="Fungal pathogen Exserohilum turcicum; long cigar-shaped lesions.",
           treatment_organic="Azoxystrobin 0.1% spray.",
           treatment_chemical="Propiconazole 25 EC foliar spray at 0.1%.",
           prevention="Use resistant hybrids. Plow under crop residues after harvest.",
           yield_loss="10–50%"),
      list(name="Maize Streak Virus (MSV)", symptoms=c("yellow_spots","curling"), severity="High",
           emoji="🟡", cause="Virus transmitted by leafhoppers (Cicadulina spp.).",
           treatment_organic="Reflective mulch to repel leafhoppers.",
           treatment_chemical="Imidacloprid seed treatment to reduce leafhopper vectors.",
           prevention="Plant early to avoid peak leafhopper seasons. Use MSV-resistant varieties.",
           yield_loss="20–100%"),
      list(name="Fall Armyworm (Spodoptera frugiperda)", symptoms=c("black_lesions","wilting"), severity="High",
           emoji="🐛", cause="Migratory moth larvae that bore into whorl and cob.",
           treatment_organic="Bacillus thuringiensis (Bt) spray inside the whorl.",
           treatment_chemical="Emamectin benzoate 5 SG at 0.4g/L spray into whorl.",
           prevention="Early scouting; apply sand+lime mixture into whorl at early infestation.",
           yield_loss="20–73%")
    ),
    Potato = list(
      list(name="Late Blight (Phytophthora infestans)", symptoms=c("water_soaked","wilting","brown_rust"), severity="High",
           emoji="🥔", cause="Most devastating potato disease; can destroy entire crop in 10 days.",
           treatment_organic="Copper hydroxide Bordeaux mixture spray every 5–7 days.",
           treatment_chemical="Metalaxyl 8% + Mancozeb 64% (Ridomil Gold MZ).",
           prevention="Plant certified seed tubers. Avoid overhead irrigation. Hill soil around stems.",
           yield_loss="Up to 100%"),
      list(name="Common Scab (Streptomyces scabies)", symptoms=c("brown_rust","black_lesions"), severity="Low",
           emoji="🟤", cause="Soil bacterium causing rough corky lesions on tuber skin.",
           treatment_organic="Maintain soil pH below 5.2 during tuber development.",
           treatment_chemical="Formalin seed tuber dip before planting.",
           prevention="Irrigate consistently during tuber formation. Use resistant varieties.",
           yield_loss="5–20% (market quality loss)"),
      list(name="Blackleg (Pectobacterium atrosepticum)", symptoms=c("black_lesions","wilting","water_soaked"), severity="High",
           emoji="⚫", cause="Bacterial disease entering through wounds; stem turns black at base.",
           treatment_organic="Remove and destroy infected plants. Improve drainage.",
           treatment_chemical="Copper-based bactericide drench.",
           prevention="Use certified healthy seed. Do not plant in waterlogged soils.",
           yield_loss="10–50%")
    ),
    Soybean = list(
      list(name="Soybean Rust (Phakopsora pachyrhizi)", symptoms=c("orange_pustules","brown_rust","yellow_spots"), severity="High",
           emoji="🟠", cause="Aggressive fungal pathogen; fastest-spreading soybean disease globally.",
           treatment_organic="Apply potassium silicate to boost resistance.",
           treatment_chemical="Tebuconazole 250 EW or Trifloxystrobin + Tebuconazole (Nativo).",
           prevention="Scout regularly. Apply fungicides preventively at R1 (flowering) stage.",
           yield_loss="10–80%"),
      list(name="Sudden Death Syndrome (SDS)", symptoms=c("yellow_spots","wilting","brown_rust"), severity="Medium",
           emoji="🟡", cause="Soil fungus Fusarium virguliforme; roots rot causing foliar symptoms.",
           treatment_organic="Trichoderma seed treatment + balanced nutrition.",
           treatment_chemical="Fluopyram seed treatment fungicide.",
           prevention="Improve soil drainage. Avoid compaction. Delayed planting in cool soils.",
           yield_loss="5–30%"),
      list(name="Bean Pod Borer (Maruca vitrata)", symptoms=c("black_lesions","wilting"), severity="Medium",
           emoji="🐛", cause="Moth larvae bore into pods and seeds during reproductive stage.",
           treatment_organic="Neem-based pesticides (NeemAzal) at pod formation.",
           treatment_chemical="Chlorpyrifos 20 EC or Monocrotophos spray.",
           prevention="Early-maturing varieties. Pheromone traps for adult moth monitoring.",
           yield_loss="10–30%")
    )
  )

  # ---- Image Preview ----
  output$ui_leaf_image_preview <- renderUI({
    f <- input$disease_image_upload
    if (is.null(f)) {
      div(
        class = "text-center p-4 bg-light rounded-3 border border-dashed mb-3",
        style = "border-style: dashed !important; border-color: #aaa !important;",
        span("🌿", style = "font-size: 3rem; display: block;"),
        div(class = "text-muted small mt-2", "Upload a leaf photo above"),
        div(class = "text-muted small", "Supported: JPG, PNG, WEBP")
      )
    } else {
      b64 <- base64enc::dataURI(file = f$datapath, mime = f$type)
      div(
        class = "text-center mb-3",
        tags$img(src = b64, style = "max-width: 100%; max-height: 220px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); object-fit: cover;"),
        div(class = "small text-muted mt-2", paste0("📁 ", f$name, " (", round(f$size/1024,1), " KB)"))
      )
    }
  })

  # ---- Disease Analysis Logic ----
  disease_analysis_rv <- reactiveVal(NULL)

  observeEvent(input$btn_analyze_disease, {
    crop     <- input$disease_crop_select
    symptoms <- input$disease_symptoms
    db       <- disease_kb[[crop]]

    if (is.null(db) || length(db) == 0) {
      disease_analysis_rv(list(type = "no_data", crop = crop))
      return()
    }

    # Score each disease by symptom overlap
    scores <- sapply(db, function(d) {
      if (length(symptoms) == 0) return(0)
      length(intersect(symptoms, d$symptoms)) / length(d$symptoms)
    })

    best_idx   <- which.max(scores)
    best_score <- scores[best_idx]

    if (length(symptoms) == 0 || best_score == 0) {
      disease_analysis_rv(list(type = "healthy", crop = crop))
    } else if (best_score >= 0.4) {
      matched   <- db[[best_idx]]
      confidence <- round(min(95, 45 + best_score * 55))
      disease_analysis_rv(list(
        type       = "match",
        crop       = crop,
        disease    = matched,
        confidence = confidence,
        score      = best_score,
        n_symptoms = length(symptoms)
      ))
    } else {
      disease_analysis_rv(list(type = "healthy", crop = crop))
    }
  })

  # ---- Render Disease Analysis Result ----
  output$ui_disease_analysis_result <- renderUI({
    res <- disease_analysis_rv()

    if (is.null(res)) {
      div(
        class = "card-modern",
        div(class = "card-header", span("🔬 Analysis Result", class = "fw-bold")),
        div(class = "p-5 text-center text-muted",
          span("🌿", style = "font-size: 3rem; display: block; margin-bottom: 12px;"),
          h5("Select symptoms and click"),
          strong("Analyse Symptoms", class = "text-danger fs-5"),
          p(class = "mt-2 small", "The system will match your observations against the disease knowledge base.")
        )
      )

    } else if (res$type == "healthy") {
      div(
        class = "disease-result-healthy",
        div(class = "text-center mb-3",
          span("✅", style = "font-size: 3.5rem; display: block;"),
          h3("No Disease Pattern Detected", class = "fw-bold text-success mt-2"),
          p(class = "text-muted", paste("Crop:", res$crop, "· Symptoms do not match any known disease profile."))
        ),
        div(class = "advisory-box",
          strong("🌱 Field Advisory: "),
          "Your crop appears healthy based on the selected symptoms. Continue regular scouting every 5–7 days. ",
          "Ensure balanced nutrition, proper irrigation scheduling, and pest monitoring."
        ),
        div(class = "small text-muted text-center mt-3",
          "⚠️ This is an educational result — always confirm with a certified plant pathologist."
        )
      )

    } else if (res$type == "match") {
      d  <- res$disease
      sev_class <- switch(d$severity, "High" = "danger", "Medium" = "warning", "Low" = "success")
      sev_icon  <- switch(d$severity, "High" = "🔴", "Medium" = "🟡", "Low" = "🟢")

      div(
        class = "disease-result-matched",
        # Header
        div(class = "d-flex justify-content-between align-items-start flex-wrap gap-2 mb-3",
          div(
            div(class = "d-flex align-items-center gap-2 mb-1",
              span(d$emoji, style = "font-size: 2rem;"),
              h4(d$name, class = "fw-bold mb-0")
            ),
            span(class = paste0("badge bg-", sev_class, " px-3 py-2"), paste(sev_icon, "Severity:", d$severity)),
            tags$small(class = "text-muted ms-2", paste("Yield Loss:", d$yield_loss))
          ),
          div(class = "text-end",
            div(class = "fs-3 fw-bold text-warning", paste0(res$confidence, "%")),
            div(class = "small text-muted", "Match confidence")
          )
        ),

        # Confidence bar
        div(class = "mb-3",
          div(class = "small fw-semibold mb-1", paste0("Symptom match: ", res$n_symptoms, " symptoms selected")),
          div(class = "progress", style = "height: 10px; border-radius: 6px;",
            div(class = paste0("progress-bar bg-", sev_class),
                style = paste0("width:", res$confidence, "%;"), role = "progressbar")
          )
        ),

        # Cause
        div(class = "mb-3",
          strong("🔬 Cause / Pathogen: "),
          span(d$cause, class = "text-muted small")
        ),

        # Treatments
        layout_columns(
          col_widths = c(6, 6),
          div(class = "advisory-box",
            strong("🌿 Organic / Natural Treatment:"),
            tags$ul(class = "small text-muted mb-0 mt-1",
              tags$li(d$treatment_organic)
            )
          ),
          div(class = "ml-advisory-box",
            strong("💊 Chemical Treatment:"),
            tags$ul(class = "small text-muted mb-0 mt-1",
              tags$li(d$treatment_chemical)
            )
          )
        ),

        div(class = "advisory-box mt-2",
          strong("🛡️ Prevention Strategy: "),
          span(d$prevention, class = "small text-muted")
        ),

        # Disclaimer
        div(
          class = "disease-disclaimer mt-3",
          div(class = "small",
            "⚠️ ", strong("Educational Result Only. "),
            "This analysis is based on reported symptoms and a reference knowledge base. ",
            "Confirm diagnosis with a certified plant pathologist before applying any pesticide. ",
            "Always follow label instructions and local agricultural department guidelines."
          )
        )
      )
    }
  })

  # ---- Disease Library Renderer ----
  output$ui_disease_library <- renderUI({
    crop <- input$disease_crop_select
    db   <- disease_kb[[crop]]
    if (is.null(db) || length(db) == 0) return(p("No diseases in database for this crop."))

    sev_css <- c("High" = "disease-severity-high", "Medium" = "disease-severity-medium", "Low" = "disease-severity-low")
    sev_badge <- c("High" = "danger", "Medium" = "warning", "Low" = "success")

    tagList(
      div(class = "mb-3",
        div(class = "d-flex align-items-center gap-2 mb-2",
          strong(paste0("Showing ", length(db), " common diseases for: ", crop))
        ),
        div(class = "d-flex gap-2 flex-wrap",
          span(class = "badge bg-danger px-3 py-1", "🔴 High Severity"),
          span(class = "badge bg-warning text-dark px-3 py-1", "🟡 Medium Severity"),
          span(class = "badge bg-success px-3 py-1", "🟢 Low Severity")
        )
      ),
      lapply(db, function(d) {
        div(
          class = paste("disease-card", sev_css[d$severity]),
          div(class = "d-flex justify-content-between align-items-start",
            div(
              div(class = "d-flex align-items-center gap-2 mb-1",
                span(d$emoji, style = "font-size: 1.4rem;"),
                strong(d$name)
              ),
              tags$small(class = "text-muted", d$cause)
            ),
            span(class = paste0("badge bg-", sev_badge[d$severity]), d$severity)
          ),
          div(class = "mt-2 d-flex gap-3 flex-wrap",
            tags$small(class = "text-danger fw-bold", paste0("⚡ Yield Loss: ", d$yield_loss)),
            tags$small(class = "text-muted", paste0("📋 Key Symptoms: ", paste(d$symptoms, collapse = ", ")))
          )
        )
      })
    )
  })

  # ============================================================================
  # PHASE 10: REPORTS & ANALYTICS SERVER LOGIC
  # ============================================================================

  # ---- KPI Outputs ----
  output$rpt_total_revenue <- renderText({
    df <- sales_val()
    if (nrow(df) > 0) paste0("₹ ", format(round(sum(df$Total_Revenue, na.rm = TRUE)), big.mark = ",")) else "₹ 0"
  })

  output$rpt_total_expenses <- renderText({
    df <- expenses_val()
    if (nrow(df) > 0) paste0("₹ ", format(round(sum(df$Amount, na.rm = TRUE)), big.mark = ",")) else "₹ 0"
  })

  output$rpt_net_profit <- renderText({
    rev <- if (nrow(sales_val()) > 0) sum(sales_val()$Total_Revenue, na.rm = TRUE) else 0
    exp <- if (nrow(expenses_val()) > 0) sum(expenses_val()$Amount, na.rm = TRUE) else 0
    p   <- rev - exp
    s   <- if (p >= 0) "+₹ " else "-₹ "
    paste0(s, format(round(abs(p)), big.mark = ","))
  })

  output$rpt_margin_text <- renderText({
    rev <- if (nrow(sales_val()) > 0) sum(sales_val()$Total_Revenue, na.rm = TRUE) else 0
    exp <- if (nrow(expenses_val()) > 0) sum(expenses_val()$Amount, na.rm = TRUE) else 0
    if (rev > 0) paste0("Profit Margin: ", round(((rev - exp) / rev) * 100, 1), "%")
    else "No sales recorded"
  })

  output$rpt_total_production <- renderText({
    df <- harvest_val()
    if (nrow(df) > 0) paste0(format(round(sum(df$Quantity_Kg, na.rm = TRUE)), big.mark = ","), " Kg")
    else "0 Kg"
  })

  # ---- Chart 1: Monthly Revenue vs Expenses Trend ----
  output$rpt_plot_monthly_trend <- renderPlotly({
    s_df <- sales_val()
    e_df <- expenses_val()

    s_monthly <- if (nrow(s_df) > 0)
      s_df %>% mutate(Month = substr(Sale_Date, 1, 7)) %>%
        group_by(Month) %>% summarise(Revenue = sum(Total_Revenue, na.rm = TRUE))
    else data.frame(Month = character(), Revenue = numeric())

    e_monthly <- if (nrow(e_df) > 0)
      e_df %>% mutate(Month = substr(Date, 1, 7)) %>%
        group_by(Month) %>% summarise(Expense = sum(Amount, na.rm = TRUE))
    else data.frame(Month = character(), Expense = numeric())

    all_months <- sort(union(s_monthly$Month, e_monthly$Month))

    if (length(all_months) == 0) {
      return(plot_ly() %>% layout(title = "No data available"))
    }

    combined <- data.frame(Month = all_months) %>%
      left_join(s_monthly, by = "Month") %>%
      left_join(e_monthly, by = "Month")
    combined[is.na(combined)] <- 0

    plot_ly(combined, x = ~Month) %>%
      add_bars(y = ~Revenue, name = "Revenue", marker = list(color = "#2e7d32")) %>%
      add_bars(y = ~Expense, name = "Expenses", marker = list(color = "#e53935")) %>%
      layout(
        barmode  = "group",
        xaxis    = list(title = "Month"),
        yaxis    = list(title = "Amount (₹)"),
        legend   = list(orientation = "h", x = 0, y = -0.25),
        margin   = list(l = 60, r = 20, t = 20, b = 60),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # ---- Chart 2: Crop-wise Revenue Contribution ----
  output$rpt_plot_crop_revenue <- renderPlotly({
    df <- sales_val()
    req(nrow(df) > 0)
    by_crop <- df %>% group_by(Crop_Name) %>% summarise(Total = sum(Total_Revenue, na.rm = TRUE)) %>% arrange(desc(Total))
    plot_ly(by_crop,
      labels  = ~Crop_Name, values = ~Total,
      type    = "pie", hole = 0.52,
      marker  = list(colors = c("#1b5e20","#2e7d32","#43a047","#81c784","#f9a825","#ef6c00","#1565c0")),
      textinfo = "label+percent"
    ) %>%
      layout(
        showlegend = TRUE,
        margin = list(l = 10, r = 10, t = 10, b = 20),
        paper_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # ---- Chart 3: Expense Category Share ----
  output$rpt_plot_expense_cat <- renderPlotly({
    df <- expenses_val()
    req(nrow(df) > 0)
    by_cat <- df %>% group_by(Category) %>% summarise(Total = sum(Amount, na.rm = TRUE)) %>% arrange(Total)
    plot_ly(by_cat,
      x = ~Total, y = ~reorder(Category, Total),
      type = "bar", orientation = "h",
      marker = list(color = "#9c27b0")
    ) %>%
      layout(
        xaxis  = list(title = "Amount (₹)"),
        yaxis  = list(title = ""),
        margin = list(l = 100, r = 20, t = 20, b = 40),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # ---- Chart 4: Actual vs Expected Yield ----
  output$rpt_plot_yield_compare <- renderPlotly({
    df <- crops_val()
    req(nrow(df) > 0)
    h_df <- harvest_val()

    expected <- df %>% select(Crop_Name, Expected_Yield = Yield_Per_Acre_Kg)
    actual <- if (nrow(h_df) > 0)
      h_df %>% group_by(Crop_Name) %>% summarise(Actual_Yield = mean(Yield_Per_Acre_Kg, na.rm = TRUE))
    else data.frame(Crop_Name = character(), Actual_Yield = numeric())

    combined <- expected %>% left_join(actual, by = "Crop_Name")
    combined[is.na(combined$Actual_Yield), "Actual_Yield"] <- 0

    plot_ly(combined, x = ~Crop_Name) %>%
      add_bars(y = ~Expected_Yield, name = "Expected (kg/acre)", marker = list(color = "#42a5f5")) %>%
      add_bars(y = ~Actual_Yield,   name = "Actual (kg/acre)",   marker = list(color = "#1b5e20")) %>%
      layout(
        barmode = "group",
        xaxis   = list(title = ""),
        yaxis   = list(title = "Yield (kg/acre)"),
        legend  = list(orientation = "h", x = 0, y = -0.25),
        margin  = list(l = 60, r = 20, t = 20, b = 60),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # ---- Chart 5: Profitability by Crop ----
  output$rpt_plot_profit_by_crop <- renderPlotly({
    s_df <- sales_val()
    e_df <- expenses_val()
    req(nrow(s_df) > 0)

    rev_by_crop <- s_df %>% group_by(Crop_Name) %>% summarise(Revenue = sum(Total_Revenue, na.rm = TRUE))

    # Allocate shared expenses equally across crops
    n_crops    <- nrow(crops_val())
    shared_exp <- if (nrow(e_df) > 0) sum(e_df$Amount, na.rm = TRUE) / max(n_crops, 1) else 0

    profit_df <- rev_by_crop %>% mutate(
      Approx_Expense = shared_exp,
      Profit         = Revenue - Approx_Expense,
      Color          = ifelse(Profit >= 0, "#2e7d32", "#c62828")
    )

    plot_ly(profit_df,
      x      = ~reorder(Crop_Name, Profit),
      y      = ~Profit,
      type   = "bar",
      marker = list(color = profit_df$Color),
      text   = ~paste0("₹ ", format(round(Profit), big.mark = ",")),
      textposition = "outside"
    ) %>%
      layout(
        xaxis  = list(title = ""),
        yaxis  = list(title = "Estimated Profit (₹)"),
        margin = list(l = 70, r = 20, t = 20, b = 50),
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)"
      )
  })

  # ---- Season Summary Table ----
  output$rpt_table_summary <- renderDT({
    c_df <- crops_val()
    h_df <- harvest_val()
    s_df <- sales_val()
    req(nrow(c_df) > 0)

    actual_yields <- if (nrow(h_df) > 0)
      h_df %>% group_by(Crop_Name) %>% summarise(Harvested_Kg = sum(Quantity_Kg, na.rm = TRUE), Avg_Yield_Actual = round(mean(Yield_Per_Acre_Kg, na.rm = TRUE)))
    else data.frame(Crop_Name = character(), Harvested_Kg = numeric(), Avg_Yield_Actual = numeric())

    revenues <- if (nrow(s_df) > 0)
      s_df %>% group_by(Crop_Name) %>% summarise(Revenue = sum(Total_Revenue, na.rm = TRUE))
    else data.frame(Crop_Name = character(), Revenue = numeric())

    summary_df <- c_df %>%
      select(Crop_Name, Variety, Area_Acres, Yield_Per_Acre_Kg, Status) %>%
      left_join(actual_yields, by = "Crop_Name") %>%
      left_join(revenues,      by = "Crop_Name")

    summary_df[is.na(summary_df$Harvested_Kg), "Harvested_Kg"]       <- 0
    summary_df[is.na(summary_df$Avg_Yield_Actual), "Avg_Yield_Actual"] <- NA
    summary_df[is.na(summary_df$Revenue), "Revenue"]                  <- 0

    summary_df$Revenue_Fmt  <- paste0("₹ ", format(round(summary_df$Revenue), big.mark = ","))
    summary_df$Harvested_Fmt <- paste0(format(summary_df$Harvested_Kg, big.mark = ","), " Kg")

    datatable(
      summary_df %>% select(
        `Crop`         = Crop_Name,
        `Variety`      = Variety,
        `Area (Acres)` = Area_Acres,
        `Expected Yield (kg/acre)` = Yield_Per_Acre_Kg,
        `Actual Yield (kg/acre)`   = Avg_Yield_Actual,
        `Harvested`    = Harvested_Fmt,
        `Revenue`      = Revenue_Fmt,
        `Status`       = Status
      ),
      options  = list(pageLength = 10, dom = "Bfrtip"),
      class    = "table table-hover align-middle",
      rownames = FALSE
    )
  })

  # ---- Download Handlers ----
  output$rpt_dl_crops    <- downloadHandler(
    filename = function() paste0("crops_",    Sys.Date(), ".csv"),
    content  = function(file) write.csv(crops_val(),    file, row.names = FALSE)
  )
  output$rpt_dl_expenses <- downloadHandler(
    filename = function() paste0("expenses_", Sys.Date(), ".csv"),
    content  = function(file) write.csv(expenses_val(), file, row.names = FALSE)
  )
  output$rpt_dl_sales    <- downloadHandler(
    filename = function() paste0("sales_",    Sys.Date(), ".csv"),
    content  = function(file) write.csv(sales_val(),    file, row.names = FALSE)
  )
  output$rpt_dl_harvest  <- downloadHandler(
    filename = function() paste0("harvest_",  Sys.Date(), ".csv"),
    content  = function(file) write.csv(harvest_val(),  file, row.names = FALSE)
  )
  output$rpt_dl_soil     <- downloadHandler(
    filename = function() paste0("soil_",     Sys.Date(), ".csv"),
    content  = function(file) write.csv(soil_val(),     file, row.names = FALSE)
  )

  output$rpt_dl_full <- downloadHandler(
    filename = function() paste0("SmartHarvest_AI_Full_Farm_Report_", Sys.Date(), ".csv"),
    content  = function(file) {
      c_df <- crops_val()
      e_df <- expenses_val()
      s_df <- sales_val()
      h_df <- harvest_val()
      so_df <- soil_val()

      rev  <- if (nrow(s_df) > 0) sum(s_df$Total_Revenue, na.rm = TRUE) else 0
      exp  <- if (nrow(e_df) > 0) sum(e_df$Amount,        na.rm = TRUE) else 0

      summary_block <- data.frame(
        Section     = "FARM SUMMARY",
        Metric      = c("Report Date", "Total Crops", "Total Area (Acres)",
                        "Total Revenue (Rs)", "Total Expenses (Rs)",
                        "Net Profit (Rs)", "Total Harvested (Kg)"),
        Value       = c(as.character(Sys.Date()),
                        nrow(c_df),
                        sum(c_df$Area_Acres, na.rm = TRUE),
                        rev, exp, rev - exp,
                        if (nrow(h_df) > 0) sum(h_df$Quantity_Kg, na.rm = TRUE) else 0),
        stringsAsFactors = FALSE
      )

      # Write a combined multi-section CSV
      con <- file(file, open = "w")

      writeLines("=== SmartHarvest AI Farm Report ===", con)
      writeLines(paste0("Generated: ", Sys.time()), con)
      writeLines("", con)

      writeLines("--- FARM KPI SUMMARY ---", con)
      write.csv(summary_block, con, row.names = FALSE)
      writeLines("", con)

      writeLines("--- CROPS ---", con)
      write.csv(c_df, con, row.names = FALSE)
      writeLines("", con)

      writeLines("--- EXPENSES ---", con)
      write.csv(e_df, con, row.names = FALSE)
      writeLines("", con)

      writeLines("--- SALES ---", con)
      write.csv(s_df, con, row.names = FALSE)
      writeLines("", con)

      writeLines("--- HARVEST ---", con)
      write.csv(h_df, con, row.names = FALSE)
      writeLines("", con)

      writeLines("--- SOIL DATA ---", con)
      write.csv(so_df, con, row.names = FALSE)

      close(con)
    }
  )

}

# ------------------------------------------------------------------------------
# 5. LAUNCH APPLICATION
# ------------------------------------------------------------------------------
shinyApp(ui = ui, server = server)
