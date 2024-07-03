package org.example;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
	public static void main(String[] args) {

		Reader in=new Reader(System.in);
		int t = in.nextInt(); // Number of test cases

		StringBuilder sb=new StringBuilder();

		while (t-- > 0) {
			/*
			  do something
			 */


		}
		System.out.println(sb.toString());
	}


	static class Reader{
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

}