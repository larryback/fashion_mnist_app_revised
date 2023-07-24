# Load required libraries
library(shiny)
library(ggplot2)
library(keras)
library(tensorflow)
library(rsconnect)
library(reshape2)



# Define the UI
ui <- fluidPage(
  titlePanel("Fashion MNIST Dashboard"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("epochs", "Number of Epochs:", min = 1, max = 50, value = 15),
      actionButton("trainBtn", "Train Model")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Training History", plotOutput("historyPlot")),
        tabPanel("Test Results", plotOutput("testResultsPlot"))
      )
    )
  )
)

