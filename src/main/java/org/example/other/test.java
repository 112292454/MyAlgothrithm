package org.example.other;

import java.util.Calendar;
import java.util.Date;
import java.util.TimeZone;

public class test {
	public static void main(String[] args) {
		int zone = 9;
		Calendar calendar = Calendar.getInstance();
		calendar.setTimeZone(TimeZone.getTimeZone("GMT+" + zone+":00"));
		System.out.println(calendar);
	}


}
