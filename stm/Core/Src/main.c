/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"
#include "input_preproc.h"
#include "mnist_samples.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
static ai_handle network = AI_HANDLE_NULL;
AI_ALIGNED(AI_NETWORK_ACTIVATIONS_ALIGNMENT)
static uint8_t activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static uint8_t ai_input_u8[AI_NETWORK_IN_1_SIZE];
static uint8_t ai_output_u8[AI_NETWORK_OUT_1_SIZE];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */

  /* Initialize X-CUBE-AI network */
  {
    const ai_handle act_addr[] = { activations };
    const ai_handle wgt_addr[] = { ai_network_data_weights_get() };
    ai_error err = ai_network_create_and_init(&network, act_addr, wgt_addr);
    if (err.type != AI_ERROR_NONE) {
      char msg[96];
      snprintf(msg, sizeof(msg), "AI init failed: type=%d code=%d\r\n", err.type, err.code);
      HAL_UART_Transmit(&huart2, (uint8_t*)msg, (uint16_t)strlen(msg), HAL_MAX_DELAY);
      Error_Handler();
    }
  }

  /* Prepare AI buffer descriptors from the network */
  ai_buffer* ai_input = ai_network_inputs_get(network, NULL);
  ai_buffer* ai_output = ai_network_outputs_get(network, NULL);
  ai_output[0].data = (ai_handle)ai_output_u8;
  /* Ensure buffers have correct metadata when overriding data pointers */
  ai_input[0].format = AI_NETWORK_IN_1_FORMAT;
  ai_output[0].format = AI_NETWORK_OUT_1_FORMAT;
  {
    char msg[128];
    int n = snprintf(msg, sizeof(msg),
                     "bufs: in=%p out=%p in_size=%lu out_size=%lu\r\n",
                     ai_input[0].data, ai_output[0].data,
                     (unsigned long)AI_NETWORK_IN_1_SIZE_BYTES,
                     (unsigned long)AI_NETWORK_OUT_1_SIZE_BYTES);
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, (uint16_t)n, HAL_MAX_DELAY);
  }

  /* --- Synthetic probes to validate data path --- */
  {
    /* Probe 1: all input at zero-point */
    memset((uint8_t*)ai_input[0].data, 33, AI_NETWORK_IN_1_SIZE_BYTES);
    memset(ai_output_u8, 0xAA, AI_NETWORK_OUT_1_SIZE_BYTES);
    ai_i32 nb = ai_network_run(network, ai_input, ai_output);
    if (nb == 1) {
      char line[160];
      uint8_t out_min = 255, out_max = 0;
      for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
        uint8_t v = ai_output_u8[k];
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
      }
      int n = snprintf(line, sizeof(line), "probe_zp: out[min=%u max=%u] q=[", out_min, out_max);
      HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
      for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
        n = snprintf(line, sizeof(line), "%u%s", ai_output_u8[k], (k+1==(int)AI_NETWORK_OUT_1_SIZE)?"]\r\n":", ");
        HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
      }
    }

    /* Probe 2: zero-point everywhere, one bright pixel */
    memset((uint8_t*)ai_input[0].data, 33, AI_NETWORK_IN_1_SIZE_BYTES);
    ((uint8_t*)ai_input[0].data)[(14*28)+14] = 255; /* center */
    memset(ai_output_u8, 0xAA, AI_NETWORK_OUT_1_SIZE_BYTES);
    nb = ai_network_run(network, ai_input, ai_output);
    if (nb == 1) {
      char line[160];
      uint8_t out_min = 255, out_max = 0;
      for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
        uint8_t v = ai_output_u8[k];
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
      }
      int n = snprintf(line, sizeof(line), "probe_dot: out[min=%u max=%u] q=[", out_min, out_max);
      HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
      for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
        n = snprintf(line, sizeof(line), "%u%s", ai_output_u8[k], (k+1==(int)AI_NETWORK_OUT_1_SIZE)?"]\r\n":", ");
        HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
      }
    }
  }

  /* Run inference on at least 10 MNIST images */
  for (int i = 0; i < MNIST_NUM_SAMPLES; ++i) {
    preprocess_u8_to_u8_quant_norm(&mnist_images[i][0], ai_input_u8);
    /* Copy into framework-managed input buffer */
    memcpy((uint8_t*)ai_input[0].data, ai_input_u8, AI_NETWORK_IN_1_SIZE_BYTES);

    /* Debug: input min/max */
    uint8_t in_min = 255, in_max = 0;
    for (int k = 0; k < (int)AI_NETWORK_IN_1_SIZE; ++k) {
      uint8_t v = ai_input_u8[k];
      if (v < in_min) in_min = v;
      if (v > in_max) in_max = v;
    }

    /* Poison output buffer to detect if runtime writes to it */
    memset(ai_output_u8, 0xAA, AI_NETWORK_OUT_1_SIZE_BYTES);

    ai_i32 nbatch = ai_network_run(network, ai_input, ai_output);
    if (nbatch != 1) {
      ai_error err = ai_network_get_error(network);
      char msg[96];
      snprintf(msg, sizeof(msg), "AI run error: type=%d code=%d\r\n", err.type, err.code);
      HAL_UART_Transmit(&huart2, (uint8_t*)msg, (uint16_t)strlen(msg), HAL_MAX_DELAY);
      Error_Handler();
    }

    /* Argmax on quantized outputs (scale>0 so argmax preserved) */
    /* Read from our output buffer (runtime should have written into it) */
    const uint8_t* out_u8 = (const uint8_t*)ai_output_u8;

    /* Debug: output min/max */
    uint8_t out_min = 255, out_max = 0;
    for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
      uint8_t v = out_u8[k];
      if (v < out_min) out_min = v;
      if (v > out_max) out_max = v;
    }

    int argmax = 0;
    uint8_t vmax = out_u8[0];
    for (int k = 1; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
      if (out_u8[k] > vmax) {
        vmax = out_u8[k];
        argmax = k;
      }
    }

    char line[160];
    int n = snprintf(line, sizeof(line),
                     "Sample %d (label=%d): pred=%d, in[min=%u max=%u] out[min=%u max=%u] out=[",
                     i, mnist_labels[i], argmax, in_min, in_max, out_min, out_max);
    HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
    for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
      n = snprintf(line, sizeof(line), "%u%s", out_u8[k],
                   (k + 1 == (int)AI_NETWORK_OUT_1_SIZE) ? "]\r\n" : ", ");
      HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
    }

    /* Also print centered logits (q - zp) without floats */
    n = snprintf(line, sizeof(line), " centered=[");
    HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
    for (int k = 0; k < (int)AI_NETWORK_OUT_1_SIZE; ++k) {
      int centered = (int)out_u8[k] - 142; /* output zp */
      n = snprintf(line, sizeof(line), "%d%s", centered,
                   (k + 1 == (int)AI_NETWORK_OUT_1_SIZE) ? "]\r\n" : ", ");
      HAL_UART_Transmit(&huart2, (uint8_t*)line, (uint16_t)n, HAL_MAX_DELAY);
    }
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
