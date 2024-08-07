package org.example.other;

import java.util.ArrayList;
import java.util.List;

public class Trie {
	private int curIndex;
	private ArrayList<int[]> trie;
	private ArrayList<Boolean> isEnd;
	private char bias;

	public Trie(int charsetSize, int maxLength, char bias) {
		//trie=new int[maxLength][charsetSize];
		trie = new ArrayList<>();
		trie.add(new int[27]);
		isEnd = new ArrayList<>();
		curIndex = 1;
		this.bias = bias;
	}

	public Trie(int charsetSize, char bias) {
		this(charsetSize, (int) (1e6 + 7), bias);
	}

	public Trie() {
		this(26, 'a');
	}

	public void add(String word) {
		int p = 0;
		//p是当前循环中，与charAt i将要匹配的节点：trie[p]中，若[index]不为0，则存储了i这个位置的字母
		for (int i = 0; i < word.length(); i++) {
			int index = word.charAt(i) - bias;
			int[] temp = trie.get(p);
			if (temp[index] == 0) {
				trie.add(new int[27]);
				temp[index] = curIndex++;
			}
			p = temp[index];
		}
		isEnd.ensureCapacity(p);
		while (isEnd.size() <= p) {
			isEnd.add(false);
		}
		isEnd.set(p, true);
	}

	public boolean contains(String word) {
		int p = 0;
		//p是当前循环中，与charAt i将要匹配的节点：trie[p]中，若[index]不为0，则存储了i这个位置的字母
		for (int i = 0; i < word.length(); i++) {
			int index = word.charAt(i) - bias;
			if (trie.get(p)[index] == 0) {
				return false;
			}
			p = trie.get(p)[index];
		}
		return isEnd.get(p);
	}



	public boolean delete(String word) {
		if (!contains(word)) return false;

		//TODO
		return true;
	}

	public boolean isPrefix(String word) {
		int p = 0;
		//p是当前循环中，与charAt i将要匹配的节点：trie[p]中，若[index]不为0，则存储了i这个位置的字母
		for (int i = 0; i < word.length(); i++) {
			int index = word.charAt(i) - bias;
			if (trie.get(p)[index] == 0) {
				return false;
			}
			p = trie.get(p)[index];
		}
		return true;
	}

	/**
	 * @return
	 * @discription 注意：这里返回的，比如有he、here，那么只会返回he，即返回前缀
	 */
	public List<String> listAllWord() {
		int p = 0;
		return listDFS(new ArrayList<>(), new char[curIndex], 0, 0);
	}


	private List<String> listDFS(List<String> res, char[] now, int nowCur, int index) {
		if (isEnd.get(index)) {
			res.add(new String(now, 0, nowCur));
			return res;
		}
		int[] temp = trie.get(0);
		for (int i = 0; i < temp.length; i++) {
			temp = trie.get(index);
			if (temp[i] != 0) {
				now[nowCur] = (char) (i + bias);
				listDFS(res, now, nowCur + 1, temp[i]);
				now[nowCur] = 0;
			}
		}
		return res;
	}
}
