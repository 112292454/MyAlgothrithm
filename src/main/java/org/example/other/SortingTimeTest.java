package org.example.other;


import org.example.lib.ot;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

public class SortingTimeTest {
	public static void main(String[] args) throws FileNotFoundException {
		Integer[] sizes = {10, 100, 1000, 10000, 100000, 1000000}; // 待测试的数组大小
		int tests = 100; // 每个数组大小的测试次数
		ot.quickSort(generateRandomArray(1000000,1000000), 0, 1000000 - 1); // 调用快速排序
		FileOutputStream fileOutputStream = new FileOutputStream("output.txt");

		// 创建一个新的 PrintStream，将其作为标准输出
		PrintStream printStream = new PrintStream(fileOutputStream);
		System.setOut(printStream);
		for (int size : sizes) {
			long quickSortTime = 0; // 用于存储快速排序的总运行时间
			long mergeSortTime = 0; // 用于存储归并排序的总运行时间
			int[] randomArray = generateRandomArray(size,1000000); // 生成随机数组

			for (int i = 0; i < tests; i++) {
				randomArray = generateRandomArray(size,1000000);
				if(i==0) System.out.println(Arrays.toString(randomArray));

				long start = System.nanoTime(); // 记录开始时间
				ot.quickSort(randomArray, 0, randomArray.length - 1); // 调用快速排序
				long end = System.nanoTime(); // 记录结束时间
				quickSortTime += end - start; // 累加运行时间
				if (i==0) System.out.println(Arrays.toString(randomArray));

				randomArray = generateRandomArray(size,1000000); // 重新生成随机数组

				start = System.nanoTime();
				ot.mergeSort(randomArray, 0, randomArray.length - 1); // 调用归并排序
				end = System.nanoTime();
				mergeSortTime += end - start;
			}

			double quickSortAvgTime = (double) quickSortTime / tests / 1e6; // 计算快速排序的平均运行时间（毫秒）
			double mergeSortAvgTime = (double) mergeSortTime / tests / 1e6; // 计算归并排序的平均运行时间（毫秒）

			System.out.println("Array size: " + size); // 打印数组大小
			System.out.println("QuickSort avg time: " + quickSortAvgTime + " ms"); // 打印快速排序的平均运行时间
			System.out.println("MergeSort avg time: " + mergeSortAvgTime + " ms"); // 打印归并排序的平均运行时间
		}
	}

	// 生成随机整数并填充数组
	public static int[] generateRandomArray(int size,int bound) {
		int[] array = new int[size];
		Random random = new Random();
		for (int i = 0; i < size; i++) {
			array[i] = random.nextInt(bound);
		}
		return array;
	}
}
