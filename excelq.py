# # import random
# #
# # def reverse_engineer_data(exp_preparation, exp_effect):
# #     result = []
# #
# #     for i in range(len(exp_preparation)):
# #         experiment_prep = exp_preparation[i]
# #         effect = exp_effect[i]
# #
# #         if experiment_prep == 0 or effect == 0:  # Handle special case
# #             result.append([0, 0, 0, 0, 0])
# #             continue
# #
# #         # Step 1: Generate reasonable values for report and scheme_design
# #         # experiment_prep = report * 0.5 + scheme_design * 0.5
# #         scheme_design = random.randint(70, 100)
# #         report = round(2 * experiment_prep - scheme_design)
# #
# #         # Adjust values if they're out of bounds
# #         if report < 70 or report > 100:
# #             # Recalculate both values to ensure they're within bounds
# #             avg = experiment_prep
# #             diff = random.randint(-15, 15)  # Allow some variation while staying within bounds
# #             report = min(max(avg + diff, 70), 100)
# #             scheme_design = round(2 * experiment_prep - report)
# #
# #             # Final bounds check
# #             report = min(max(report, 70), 100)
# #             scheme_design = min(max(scheme_design, 70), 100)
# #
# #         # Verify the preparation formula
# #         calculated_prep = round(report * 0.5 + scheme_design * 0.5)
# #         if calculated_prep != experiment_prep:
# #             # Adjust if necessary
# #             report = round(2 * experiment_prep - scheme_design)
# #             report = min(max(report, 70), 100)
# #
# #         # Step 2: Generate values for completed and run_effect, then calculate algo_understanding
# #         # effect = completed * 0.25 + run_effect * 0.25 + algo_understanding * 0.5
# #         completed = random.randint(70, 100)
# #         run_effect = random.randint(70, 100)
# #
# #         # Calculate algo_understanding based on the effect formula
# #         algo_understanding = round((effect - completed * 0.25 - run_effect * 0.25) / 0.5)
# #
# #         # If algo_understanding is out of bounds, adjust the other values
# #         if algo_understanding < 70 or algo_understanding > 100:
# #             # Recalculate to ensure all values are within bounds
# #             algo_understanding = random.randint(70, 100)
# #             completed = random.randint(70, 100)
# #             run_effect = round((effect - completed * 0.25 - algo_understanding * 0.5) * 4)
# #
# #             # Final bounds check
# #             completed = min(max(completed, 70), 100)
# #             run_effect = min(max(run_effect, 70), 100)
# #             algo_understanding = min(max(algo_understanding, 70), 100)
# #
# #         # Verify the effect formula
# #         calculated_effect = round(completed * 0.25 + run_effect * 0.25 + algo_understanding * 0.5)
# #         if calculated_effect != effect:
# #             # Adjust algo_understanding if necessary
# #             algo_understanding = round((effect - completed * 0.25 - run_effect * 0.25) / 0.5)
# #             algo_understanding = min(max(algo_understanding, 70), 100)
# #
# #         result.append([report, scheme_design, completed, run_effect, algo_understanding])
# #
# #     return result
# #
# # # Test with your data
# # experiment_preparation = [85, 86, 70, 80, 80, 70, 80, 80, 90, 90, 70, 82, 90, 88, 78, 70, 90, 90, 85, 85,
# #                           80, 85, 80, 80, 85, 80, 75, 80, 82, 80, 78, 80, 90, 78, 80, 80, 82, 70, 80, 78,
# #                           85, 73, 80, 78, 75, 80, 70, 85, 83, 80, 72, 75, 80, 89, 90, 82, 80, 85, 80, 78,
# #                           82, 78, 83, 91, 88, 90, 92, 90, 89, 86, 82, 80, 78, 0, 86, 78, 82, 80]
# # experiment_effect = [85, 90, 70, 80, 82, 83, 88, 82, 92, 90, 72, 85, 89, 88, 90, 72, 90, 91, 93, 90,
# #                      80, 83, 80, 81, 85, 85, 80, 80, 80, 78, 82, 88, 88, 80, 72, 90, 80, 75, 90, 75,
# #                      80, 75, 78, 76, 80, 78, 78, 91, 92, 88, 88, 86, 82, 89, 88, 85, 80, 91, 85, 88,
# #                      84, 78, 80, 88, 90, 90, 90, 90, 89, 75, 87, 78, 78, 0, 86, 78, 90, 80]
# #
# # # Generate and print results
# # generated_data = reverse_engineer_data(experiment_preparation, experiment_effect)
# # print("Format: [预习报告, 方案设计, 完成度, 运行效果, 算法理解]")
# # for row in generated_data:
# #     print(row)
# import random
#
#
# def reverse_engineer_data(exp_preparation, exp_effect):
#     result = []
#
#     for i in range(len(exp_preparation)):
#         experiment_prep = exp_preparation[i]
#         effect = exp_effect[i]
#
#         if experiment_prep == 0 or effect == 0:  # Handle special case
#             result.append([0, 0, 0, 0, 0])
#             continue
#
#         # Step 1: 确保预习报告和方案设计的平均值等于实验准备
#         tries = 0
#         while tries < 100:
#             scheme_design = random.randint(70, 100)
#             report = 2 * experiment_prep - scheme_design
#             if 70 <= report <= 100:
#                 # 验证平均值是否精确等于实验准备
#                 if (report + scheme_design) // 2 == experiment_prep:
#                     break
#             tries += 1
#
#         # 如果没找到合适的值，强制设定
#         if tries == 100:
#             report = experiment_prep
#             scheme_design = experiment_prep
#
#         # Step 2: 确保完成度、运行效果和算法理解的加权平均等于实验效果
#         tries = 0
#         while tries < 100:
#             completed = random.randint(70, 100)
#             run_effect = random.randint(70, 100)
#             # 确保算法理解的计算结果为整数
#             total = effect * 4  # 将效果乘以4以避免小数
#             remaining = total - completed - run_effect
#             algo_understanding = remaining // 2  # 除以2因为算法理解权重是0.5
#
#             # 验证是否满足条件
#             if 70 <= algo_understanding <= 100:
#                 calculated_effect = (completed + run_effect + algo_understanding * 2) // 4
#                 if calculated_effect == effect:
#                     break
#             tries += 1
#
#         # 如果没找到合适的值，强制设定
#         if tries == 100:
#             completed = effect
#             run_effect = effect
#             algo_understanding = effect
#
#         result.append([report, scheme_design, completed, run_effect, algo_understanding])
#
#     return result
#
#
# # Test with your data
# experiment_preparation = [85, 86, 70, 80, 80, 70, 80, 80, 90, 90, 70, 82, 90, 88, 78, 70, 90, 90, 85, 85,
#                           80, 85, 80, 80, 85, 80, 75, 80, 82, 80, 78, 80, 90, 78, 80, 80, 82, 70, 80, 78,
#                           85, 73, 80, 78, 75, 80, 70, 85, 83, 80, 72, 75, 80, 89, 90, 82, 80, 85, 80, 78,
#                           82, 78, 83, 91, 88, 90, 92, 90, 89, 86, 82, 80, 78, 0, 86, 78, 82, 80]
# experiment_effect = [85, 90, 70, 80, 82, 83, 88, 82, 92, 90, 72, 85, 89, 88, 90, 72, 90, 91, 93, 90,
#                      80, 83, 80, 81, 85, 85, 80, 80, 80, 78, 82, 88, 88, 80, 72, 90, 80, 75, 90, 75,
#                      80, 75, 78, 76, 80, 78, 78, 91, 92, 88, 88, 86, 82, 89, 88, 85, 80, 91, 85, 88,
#                      84, 78, 80, 88, 90, 90, 90, 90, 89, 75, 87, 78, 78, 0, 86, 78, 90, 80]
#
# # Generate and print results
# generated_data = reverse_engineer_data(experiment_preparation, experiment_effect)
# print("Format: [预习报告, 方案设计, 完成度, 运行效果, 算法理解]")
# for row in generated_data:
#     print(row)
#
# # 验证结果
# print("\n验证结果:")
# for i in range(len(generated_data)):
#     prep = generated_data[i]
#     calculated_prep = (prep[0] + prep[1]) // 2
#     calculated_effect = (prep[2] + prep[3] + prep[4] * 2) // 4
#
#     if calculated_prep != experiment_preparation[i] or calculated_effect != experiment_effect[i]:
#         print(f"错误：索引 {i}")
#         print(f"期望的实验准备: {experiment_preparation[i]}, 计算得到: {calculated_prep}")
#         print(f"期望的实验效果: {experiment_effect[i]}, 计算得到: {calculated_effect}")

import random


def reverse_engineer_data(exp_preparation, exp_effect):
    result = []

    for i in range(len(exp_preparation)):
        experiment_prep = exp_preparation[i]
        effect = exp_effect[i]

        if experiment_prep == 0 or effect == 0:  # 处理特殊情况
            result.append([0, 0, 0, 0, 0])
            continue

        # Step 1: 生成预习报告和方案设计，使其平均值等于实验准备
        tries = 0
        while tries < 100:
            scheme_design = random.randint(70, 100)
            report = 2 * experiment_prep - scheme_design
            if 70 <= report <= 100 and (report + scheme_design) // 2 == experiment_prep:
                break
            tries += 1

        if tries == 100:
            report, scheme_design = experiment_prep, experiment_prep

        # Step 2: 生成完成度、运行效果和算法理解，使其尽量接近，并满足实验效果
        base_value = effect + random.randint(-2, 2)  # 让三个值接近 effect
        completed = min(100, max(70, base_value + random.randint(-2, 2)))
        run_effect = min(100, max(70, base_value + random.randint(-2, 2)))
        algo_understanding = min(100, max(70, (effect * 4 - completed - run_effect) // 2))

        # 校正：确保最终计算出的 effect 符合预期
        calculated_effect = (completed + run_effect + algo_understanding * 2) // 4
        if calculated_effect != effect:
            algo_understanding = min(100, max(70, algo_understanding + (effect - calculated_effect)))

        # 最终检查
        if (completed + run_effect + algo_understanding * 2) // 4 != effect:
            completed = run_effect = algo_understanding = effect

        result.append([report, scheme_design, completed, run_effect, algo_understanding])

    return result



test_preparation = [85, 86, 70, 80, 80, 70, 80, 80, 90, 90, 70, 82, 90, 88, 78, 70, 90, 90, 85, 85,
                          80, 85, 80, 80, 85, 80, 75, 80, 82, 80, 78, 80, 90, 78, 80, 80, 82, 70, 80, 78,
                          85, 73, 80, 78, 75, 80, 70, 85, 83, 80, 72, 75, 80, 89, 90, 82, 80, 85, 80, 78,
                          82, 78, 83, 91, 88, 90, 92, 90, 89, 86, 82, 80, 78, 0, 86, 78, 82, 80]
test_effect = [85, 90, 70, 80, 82, 83, 88, 82, 92, 90, 72, 85, 89, 88, 90, 72, 90, 91, 93, 90,
                     80, 83, 80, 81, 85, 85, 80, 80, 80, 78, 82, 88, 88, 80, 72, 90, 80, 75, 90, 75,
                     80, 75, 78, 76, 80, 78, 78, 91, 92, 88, 88, 86, 82, 89, 88, 85, 80, 91, 85, 88,
                     84, 78, 80, 88, 90, 90, 90, 90, 89, 75, 87, 78, 78, 0, 86, 78, 90, 80]

generated_data = reverse_engineer_data(test_preparation, test_effect)
print("[预习报告, 方案设计, 完成度, 运行效果, 算法理解]")
i=1
for row in generated_data:

    print(i,row)
    i += 1
