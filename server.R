# Load required libraries
library(shiny)
library(ggplot2)
library(keras)
library(tensorflow)
library(rsconnect)
library(reshape2)




# Define the server
server <- function(input, output) {
  # Print a message to the log when the application starts
  cat("Application started\n")
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