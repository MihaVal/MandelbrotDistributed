import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;
import javax.imageio.ImageIO;
import mpi.*;

public class MandelbrotViewer extends JFrame {
    private int width = 800;
    private int height = 600;
    private static final int MAX_ITER = 100;
    private static final int TARGET_FPS = 60;

    private double xMin = -2.0, xMax = 1.0;
    private double yMin = -1.2, yMax = 1.2;
    private double zoomFactor = 0.8;
    private double panSpeed;
    private BufferedImage image;
    private volatile boolean rendering = false; // volatile for thread safety

    // Command-line flags
    private static boolean headlessMode = false;
    private static boolean runTests = false;

    // MPI specific fields
    private final int rank;
    private final int size;

    public MandelbrotViewer(int rank, int size) {
        this.rank = rank;
        this.size = size;

        // Only the master process (rank 0) initializes the GUI
        if (rank == 0 && !headlessMode && !runTests) {
            initGUI();
        }

        image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        updatePanSpeed();

        // This is called by the master on startup
        if (rank == 0) {
            startRendering();
        }
    }

    private void initGUI() {
        setTitle("Mandelbrot-MPI Window (Master Rank 0)");
        setSize(width, height);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE); // We handle closing to shutdown MPI
        setLocationRelativeTo(null);
        setResizable(true);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                // This is the trigger for a graceful shutdown
                shutdown();
            }
        });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent ev) {
                switch (ev.getKeyCode()) {
                    case KeyEvent.VK_LEFT:  xMin -= panSpeed; xMax -= panSpeed; break;
                    case KeyEvent.VK_RIGHT: xMin += panSpeed; xMax += panSpeed; break;
                    case KeyEvent.VK_UP:    yMin -= panSpeed; yMax -= panSpeed; break;
                    case KeyEvent.VK_DOWN:  yMin += panSpeed; yMax += panSpeed; break;
                    case KeyEvent.VK_1: zoom(zoomFactor); break;
                    case KeyEvent.VK_2: zoom(1 / zoomFactor); break;
                    case KeyEvent.VK_S: promptAndSaveImage(); break;
                    case KeyEvent.VK_Q: shutdown(); break; // Explicit quit key
                }
                startRendering();
            }
        });

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                width = getWidth();
                height = getHeight();
                startRendering();
            }
        });
    }



    private void updatePanSpeed() {
        panSpeed = 0.1 * (xMax - xMin);
    }

    private void zoom(double factor) {
        double xCenter = (xMin + xMax) / 2;
        double yCenter = (yMin + yMax) / 2;
        double xRange = (xMax - xMin) * factor;
        double yRange = (yMax - yMin) * factor;
        xMin = xCenter - xRange / 2;
        xMax = xCenter + xRange / 2;
        yMin = yCenter - yRange / 2;
        yMax = yCenter + yRange / 2;
        updatePanSpeed();
    }

    private void startRendering() {
        if (rendering || rank != 0) return; // Only master initiates a render

        rendering = true;

        // **CRITICAL FIX**: Run the blocking MPI render logic on a background thread.
        // This prevents the GUI's Event Dispatch Thread (EDT) from freezing.
        new Thread(() -> {
            renderMandelbrot(this.width, this.height);
            rendering = false;
            if (!headlessMode && !runTests) {
                // repaint() is thread-safe and will schedule a paint on the EDT.
                repaint();
            }
        }).start();
    }

    // Master's orchestration logic for a render of specific dimensions
    private void renderMandelbrot(int renderWidth, int renderHeight) {
        double[] params = new double[]{renderWidth, renderHeight, xMin, xMax, yMin, yMax};
        try {
            MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);
            BufferedImage resultImage = computeAndGather(renderWidth, renderHeight);
            if (rank == 0) {
                this.image = resultImage;
            }
        } catch (MPIException e) {
            e.printStackTrace();
        }
    }

    // This method is run by ALL processes after receiving parameters.
    private BufferedImage computeAndGather(int renderWidth, int renderHeight) throws MPIException {
        long start = 0;
        if (rank == 0) start = System.currentTimeMillis();

        int rowsPerProc = renderHeight / size;
        int extraRows = renderHeight % size;
        int yStart = rank * rowsPerProc + Math.min(rank, extraRows);
        int yEnd = yStart + rowsPerProc + (rank < extraRows ? 1 : 0);

        int[] localPixels = new int[(yEnd - yStart) * renderWidth];
        int idx = 0;
        for (int y = yStart; y < yEnd; y++) {
            for (int x = 0; x < renderWidth; x++) {
                double real = xMin + x * (xMax - xMin) / renderWidth;
                double imag = yMin + y * (yMax - yMin) / renderHeight;
                localPixels[idx++] = computePoint(new Complex(real, imag));
            }
        }

        int[] recvCounts = (rank == 0) ? new int[size] : null;
        int[] displs = (rank == 0) ? new int[size] : null;

        if (rank == 0) {
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int startRow = i * rowsPerProc + Math.min(i, extraRows);
                int endRow = startRow + rowsPerProc + (i < extraRows ? 1 : 0);
                recvCounts[i] = (endRow - startRow) * renderWidth;
                displs[i] = offset;
                offset += recvCounts[i];
            }
        }

        int[] allPixels = (rank == 0) ? new int[renderWidth * renderHeight] : null;
        MPI.COMM_WORLD.Gatherv(localPixels, 0, localPixels.length, MPI.INT,
                allPixels, 0, recvCounts, displs, MPI.INT, 0);

        BufferedImage finalImage = null;
        if (rank == 0) {
            finalImage = new BufferedImage(renderWidth, renderHeight, BufferedImage.TYPE_INT_RGB);
            finalImage.setRGB(0, 0, renderWidth, renderHeight, allPixels, 0, renderWidth);
            long end = System.currentTimeMillis();
            System.out.println("INFO: MPI Render (" + renderWidth + "x" + renderHeight + " on " + size + " procs): " + (end - start) + " ms");
        }
        return finalImage;
    }

    private int computePoint(Complex c) {
        Complex z = new Complex(0, 0);
        int n = 0;
        while (z.abs() <= 2 && n < MAX_ITER) {
            z = z.multiply(z).add(c);
            n++;
        }
        if (n == MAX_ITER) return Color.BLACK.getRGB();
        return Color.HSBtoRGB(0.7f + (float) n / MAX_ITER, 1.0f, 1.0f);
    }

    @Override
    public void paint(Graphics g) {
        // The super.paint() is important to clear the panel before drawing
        super.paint(g);
        if (rank == 0) {
            g.drawImage(image, 0, 0, this);
        }
    }

    // This method initiates the graceful shutdown of all MPI processes.
    private void shutdown() {
        if (rank == 0) {
            System.out.println("INFO: Master sending shutdown signal...");
            double[] quitSignal = new double[]{-1.0, 0, 0, 0, 0, 0};
            try {
                // Tell workers to exit their loops
                MPI.COMM_WORLD.Bcast(quitSignal, 0, 6, MPI.DOUBLE, 0);
            } catch (MPIException e) {
                e.printStackTrace();
            }
            dispose(); // Close the GUI window
        }
    }

    // Main loop for worker processes (ranks > 0).
    public void runWorker() {
        if (rank == 0) return;

        while (true) {
            try {
                double[] params = new double[6];
                MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);

                if (params[0] < 0) { // Check for the shutdown signal
                    System.out.println("INFO: Worker " + rank + " received shutdown signal. Exiting loop.");
                    break;
                }

                int renderWidth = (int) params[0];
                int renderHeight = (int) params[1];
                this.xMin = params[2];
                this.xMax = params[3];
                this.yMin = params[4];
                this.yMax = params[5];

                computeAndGather(renderWidth, renderHeight);
            } catch (MPIException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    // In MandelbrotViewer class

    private static void runPerformanceTests(MandelbrotViewer viewer) {
        int startSize = 1000;
        int maxSize = 10000;
        String csvFile = "mandelbrot_mpi_results.csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            writer.println("width,height,distributed");

            for (int size = startSize; size <= maxSize; size += 1000) {
                // Start the timer
                long startTime = System.currentTimeMillis();

                // Run the blocking render operation
                viewer.renderMandelbrot(size, size);

                // Stop the timer
                long endTime = System.currentTimeMillis();
                long renderTime = endTime - startTime;

                // Log to console and write to file
                System.out.println("STATUS: Test render " + size + "x" + size + " completed in " + renderTime + " ms");
                writer.println(size + "," + size + "," + renderTime);
            }
            System.out.println("SUCCESS: Performance tests complete. Results saved to " + csvFile);
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("ERROR: Could not write to CSV file.");
        }
    }

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        for (String arg : args) {
            if (arg.equalsIgnoreCase("--nongui")) headlessMode = true;
            else if (arg.equalsIgnoreCase("--test")) runTests = true;
        }

        MandelbrotViewer viewer = new MandelbrotViewer(rank, size);

        if (rank == 0) {
            // --- MASTER (Rank 0) LOGIC ---
            if (runTests) {
                runPerformanceTests(viewer);
                viewer.shutdown();
            } else if (headlessMode) {
                // The main thread will block here until the background render is done.
                // We need a way to wait. A simple but crude way:
                while(viewer.rendering) { try { Thread.sleep(100); } catch(InterruptedException e){} }
                viewer.shutdown();
            } else {
                // GUI Mode: Let the Event Dispatch Thread manage the application's life.
                // The master's main thread will proceed to MPI.Finalize() and block there,
                // waiting for workers. This is the correct behavior.
                SwingUtilities.invokeLater(() -> viewer.setVisible(true));
            }
        } else {
            // --- WORKER (Rank > 0) LOGIC ---
            viewer.runWorker(); // This blocks until a shutdown signal is received.
        }

        // All processes will reach here eventually.
        // Master blocks here until workers are done. Workers block until they get a quit signal.
        MPI.Finalize();
    }

    // Unchanged Complex class
    static class Complex {
        private final double real, imag;
        public Complex(double real, double imag) { this.real = real; this.imag = imag; }
        public Complex add(Complex other) { return new Complex(this.real + other.real, this.imag + other.imag); }
        public Complex multiply(Complex other) { return new Complex(this.real * other.real - this.imag * other.imag, this.real * other.imag + this.imag * other.real); }
        public double abs() { return Math.sqrt(real * real + imag * imag); }
    }
}