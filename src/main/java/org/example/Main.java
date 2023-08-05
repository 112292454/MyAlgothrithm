package org.example;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		int t = scanner.nextInt(); // Number of test cases

		StringBuilder sb=new StringBuilder();

		while (t-- > 0) {
			ArrayList<score> al=new ArrayList<>();
			int n = scanner.nextInt(),m= scanner.nextInt(),T=scanner.nextInt(); // Number of people
			for (int j = 0; j < n; j++) {
				int[] person=new int[m];
				for (int i = 0; i < m; i++) {
					person[i]=scanner.nextInt();
				}
				Arrays.sort(person);
				int sum=0,used=0;
				int i = 0;
				for (; i < person.length; i++) {
					sum+=person[i]+used;
					used+=person[i];
					if(used >T) break;
				}
				al.add(new score(i,sum-(i>0?person[i-1]:0)));
			}
			/*al.forEach(a->{
				System.out.println(a.k+" "+a.time);
			});*/

			AtomicInteger cnt= new AtomicInteger(1);
			score s = al.get(0);
			al.remove(0);
			al.forEach(a-> {
				if (s.k <= a.k) {
					if (s.k < a.k) cnt.getAndIncrement();
					else if (s.time > a.time) cnt.getAndIncrement();
				}
			});
			sb.append(cnt).append("\n");
		}
		System.out.println(sb.toString());
	}

	static class score{
		int k;
		int time;

		public score(int k, int time) {
			this.k = k;
			this.time = time;
		}
	}
}
class Reader{
	private final static int BUF_SZ = 65536;
	BufferedReader in;
	StringTokenizer tokenizer;
	public Reader(InputStream in) {
		super();
		this.in = new BufferedReader(new InputStreamReader(in),BUF_SZ);
		tokenizer = new StringTokenizer("");
	}

	public String next() {
		while (!tokenizer.hasMoreTokens()) {
			try {
				tokenizer = new StringTokenizer(in.readLine());
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken();
	}
	public int nextInt() {
		return Integer.parseInt(next());
	}
	public long nextLong() {
		return Long.parseLong(next());
	}
}
