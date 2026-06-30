/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.cpp
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
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
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "arm_math.h"
#include "dsp/filtering_functions.h"
#include "hgag_classifier.h"
#include "stm32l4xx_hal_def.h"
#include "stm32l4xx_hal_i2c.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <cstdint>
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
I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim2;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

namespace {
using HgagOpResolver = tflite::MicroMutableOpResolver<7>;

TfLiteStatus RegisterOps(HgagOpResolver &op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExpandDims());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  return kTfLiteOk;
}

const float32_t filter_coeffs[20] = {
    0.16717926860848994f,
    -0.3343585372169799f,
    0.16717926860848994f,
    0.5418421547938043f,
    -0.12428811601288457f,
    1.0f,
    2.0f,
    1.0f,
    -1.4203516303076948f,
    -0.5196607446342449f,
    1.0f,
    -2.0f,
    1.0f,
    0.9066302244462777f,
    -0.5744851222349661f,
    1.0f,
    2.0f,
    1.0f,
    -1.7198734191690779f,
    -0.811727845882331f,
};

float32_t accel_x_state[8];
float32_t accel_y_state[8];
float32_t accel_z_state[8];
float32_t gyro_x_state[8];
float32_t gyro_y_state[8];
float32_t gyro_z_state[8];

arm_biquad_cascade_df2T_instance_f32 accel_x_filter;
arm_biquad_cascade_df2T_instance_f32 accel_y_filter;
arm_biquad_cascade_df2T_instance_f32 accel_z_filter;
arm_biquad_cascade_df2T_instance_f32 gyro_x_filter;
arm_biquad_cascade_df2T_instance_f32 gyro_y_filter;
arm_biquad_cascade_df2T_instance_f32 gyro_z_filter;

} // namespace

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM2_Init(void);
static void MX_I2C1_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick.
   */
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
  MX_TIM2_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */

  arm_biquad_cascade_df2T_init_f32(&accel_x_filter, 4, filter_coeffs,
                                   accel_x_state);
  arm_biquad_cascade_df2T_init_f32(&accel_y_filter, 4, filter_coeffs,
                                   accel_y_state);
  arm_biquad_cascade_df2T_init_f32(&accel_z_filter, 4, filter_coeffs,
                                   accel_z_state);
  arm_biquad_cascade_df2T_init_f32(&gyro_x_filter, 4, filter_coeffs,
                                   gyro_x_state);
  arm_biquad_cascade_df2T_init_f32(&gyro_y_filter, 4, filter_coeffs,
                                   gyro_y_state);
  arm_biquad_cascade_df2T_init_f32(&gyro_z_filter, 4, filter_coeffs,
                                   gyro_z_state);

  uint8_t who_am_i = 0;
  HAL_I2C_Mem_Read(&hi2c1, 0x68 << 1, 0x75, I2C_MEMADD_SIZE_8BIT, &who_am_i, 1,
                   HAL_MAX_DELAY);

  char buf[64];
  HAL_StatusTypeDef status =
      HAL_I2C_Mem_Read(&hi2c1, 0x68 << 1, 0x75, I2C_MEMADD_SIZE_8BIT, &who_am_i,
                       1, HAL_MAX_DELAY);
  snprintf(buf, sizeof(buf), "WHO_AM_I: 0x%02X, status: %d\r\n", who_am_i,
           status);
  HAL_UART_Transmit(&huart2, reinterpret_cast<uint8_t *>(buf), strlen(buf),
                    HAL_MAX_DELAY);

  uint8_t wake_data = 0x00;
  HAL_StatusTypeDef wake_status =
      HAL_I2C_Mem_Write(&hi2c1, 0x68 << 1, 0x6B, I2C_MEMADD_SIZE_8BIT,
                        &wake_data, 1, HAL_MAX_DELAY);
  snprintf(buf, sizeof(buf), "wake: %d\r\n", wake_status);
  HAL_UART_Transmit(&huart2, reinterpret_cast<uint8_t *>(buf), strlen(buf),
                    HAL_MAX_DELAY);

  uint8_t data = 0x04;
  HAL_I2C_Mem_Write(&hi2c1, 0x68 << 1, 0x19, I2C_MEMADD_SIZE_8BIT, &data, 1,
                    HAL_MAX_DELAY);

  uint8_t dlpf_data = 0x01;
  HAL_I2C_Mem_Write(&hi2c1, 0x68 << 1, 0x1A, I2C_MEMADD_SIZE_8BIT, &dlpf_data,
                    1, HAL_MAX_DELAY);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {
    uint8_t raw_accel[6];
    status = HAL_I2C_Mem_Read(&hi2c1, 0x68 << 1, 0x3B, I2C_MEMADD_SIZE_8BIT,
                              raw_accel, 6, HAL_MAX_DELAY);

    float accel_x = (int16_t)(raw_accel[0] << 8 | raw_accel[1]);
    float accel_y = (int16_t)(raw_accel[2] << 8 | raw_accel[3]);
    float accel_z = (int16_t)(raw_accel[4] << 8 | raw_accel[5]);

    accel_x /= 16384.0;
    accel_y /= 16384.0;
    accel_z /= 16384.0;

    arm_biquad_cascade_df2T_f32(&accel_x_filter, &accel_x, &accel_x, 1);
    arm_biquad_cascade_df2T_f32(&accel_y_filter, &accel_y, &accel_y, 1);
    arm_biquad_cascade_df2T_f32(&accel_z_filter, &accel_z, &accel_z, 1);

    snprintf(buf, sizeof(buf),
             "ACCEL -- X: %.4f, Y: %.4f, Z: %.4f, status: %d\r\n", accel_x,
             accel_y, accel_z, status);
    HAL_UART_Transmit(&huart2, reinterpret_cast<uint8_t *>(buf), strlen(buf),
                      HAL_MAX_DELAY);

    uint8_t raw_gyro[6];
    status = HAL_I2C_Mem_Read(&hi2c1, 0x68 << 1, 0x43, I2C_MEMADD_SIZE_8BIT,
                              raw_gyro, 6, HAL_MAX_DELAY);

    float gyro_x = (int16_t)(raw_gyro[0] << 8 | raw_gyro[1]);
    float gyro_y = (int16_t)(raw_gyro[2] << 8 | raw_gyro[3]);
    float gyro_z = (int16_t)(raw_gyro[4] << 8 | raw_gyro[5]);

    gyro_x /= 131.0;
    gyro_y /= 131.0;
    gyro_z /= 131.0;

    arm_biquad_cascade_df2T_f32(&gyro_x_filter, &gyro_x, &gyro_x, 1);
    arm_biquad_cascade_df2T_f32(&gyro_y_filter, &gyro_y, &gyro_y, 1);
    arm_biquad_cascade_df2T_f32(&gyro_z_filter, &gyro_z, &gyro_z, 1);

    snprintf(buf, sizeof(buf),
             "GYRO -- X: %.4f, Y: %.4f, Z: %.4f, status: %d\r\n", gyro_x,
             gyro_y, gyro_z, status);
    HAL_UART_Transmit(&huart2, reinterpret_cast<uint8_t *>(buf), strlen(buf),
                      HAL_MAX_DELAY);
    HAL_Delay(5);

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK) {
    Error_Handler();
  }

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
    Error_Handler();
  }
}

/**
 * @brief I2C1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_I2C1_Init(void) {

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x10D19CE4;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
    Error_Handler();
  }

  /** Configure Analogue filter
   */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK) {
    Error_Handler();
  }

  /** Configure Digital filter
   */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */
}

/**
 * @brief TIM2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM2_Init(void) {

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 79;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK) {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK) {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);
}

/**
 * @brief USART2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART2_UART_Init(void) {

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
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK) {
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
static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1) {
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
void assert_failed(uint8_t *file, uint32_t line) {
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line
     number, ex: printf("Wrong parameters value: file %s on line %d\r\n", file,
     line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
