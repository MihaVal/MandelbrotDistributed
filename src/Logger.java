import java.text.SimpleDateFormat;

public class Logger {
    private final static String Reset = "\u001b[0m";
    private final static String Red = "\u001b[31m";
    private final static String Green = "\u001b[32m";
    private final static String Yellow = "\u001b[33m";
    private final static String Blue = "\u001b[34m";
    private final static String Magenta = "\u001b[35m";
    private final static String Cyan =  "\u001b[36m";
    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");


    public static void log(String message) {
        log(message, LogLevel.Info);
    }

    public static void log(String message, LogLevel level) {
        String dateString = dateFormat.format(System.currentTimeMillis());
        String threadName = Thread.currentThread().getName();

        String messagePrefix = "[" + dateString + "][" + threadName + "] " + level + ": ";
        // [2024-03-03 21:45:23.344][threadName] Info: naš message

        switch (level){
            case Info -> messagePrefix = Yellow + messagePrefix + Reset;
            case Debug -> messagePrefix = Magenta + messagePrefix + Reset;
            case Warn -> messagePrefix = Cyan + messagePrefix + Reset;
            case Error -> messagePrefix = Red + messagePrefix + Reset;
            case Success -> messagePrefix = Green + messagePrefix + Reset;
            case Status -> messagePrefix = Blue + messagePrefix + Reset;
        }

        System.out.println(messagePrefix + message);
    }
}
