options(browser = "C:/Program Files/Google/Chrome/Application/chrome.exe")
setwd("C:/R/R_folder/app_folder-2/fashion_mnist_app")
getwd()



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

# Define the server
server <- function(input, output) {
  # Import dataset
  fashion_mnist <- dataset_fashion_mnist()
  c(train_images, train_labels) %<-% fashion_mnist$train
  c(test_images, test_labels) %<-% fashion_mnist$test
  
  # Normalize the data
  train_images <- train_images / 255
  test_images <- test_images / 255
  
  # Reshape the data
  x_train <- train_images %>%
    array_reshape(c(60000, 28, 28, 1))
  x_test <- test_images %>%
    array_reshape(c(10000, 28, 28, 1))
  
  # Define the class names
  class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
  
  # Train the model
  model <- NULL
  observeEvent(input$trainBtn, {
    model <<- keras_model_sequential()
    model %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_flatten() %>%
      layer_dense(units = 128, activation = 'relu') %>%
      layer_dropout(rate = 0.5) %>%
      layer_dense(units = 10, activation = 'softmax')
    
    model %>% compile(
      loss = 'sparse_categorical_crossentropy',
      optimizer = 'adam', 
      metrics = c('accuracy'))
    
    history <- model %>% fit(x_train, train_labels, epochs = input$epochs, verbose = 2)
    
    output$historyPlot <- renderPlot({
      plot(history)
    })
  })
  
  # Evaluate the model on test data
  output$testResultsPlot <- renderPlot({
    if (!is.null(model)) {
      predictions <- model %>% predict(x_test)
      
      df <- data.frame(
        TrueLabel = class_names[test_labels + 1],
        PredictedLabel = class_names[apply(predictions, 1, which.max) + 1]
      )
      
      ggplot(df, aes(x = TrueLabel, fill = PredictedLabel)) +
        geom_bar() +
        labs(title = "Test Results", x = "True Label", y = "Count") +
        theme_minimal() +
        theme(legend.position = "right")
    }
  })
}


# Run the app

shinyApp(ui = ui, server = server)


# Add deployment configuration
rsconnect::setAccountInfo(name='insightfulapps-for-everyone',
                          token='16999D937A4EE185888CE7751C7E1B71',
                          secret='kusXTkNfLTZ/kHdOS5TxsQrgBH5Td4o5P5t3vZM2')


#rsconnect::deployApp("C:/R/R_folder/app_folder-2/fashion_mnist_app")

#rsconnect::showLogs("C:/R/R_folder/app_folder-2/fashion_mnist_app")

#library(tools)
#file_path_as_absolute(fashion_mnist_classification.R)

