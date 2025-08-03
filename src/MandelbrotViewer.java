import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import javax.imageio.ImageIO;
import mpi.*;

public class MandelbrotViewer extends JFrame {
    private int width = 800;
    private int height = 600;
    private static final int MAX_ITER = 250;
    private double xMin = -2.0, xMax = 1.0;
    private double yMin = -1.2, yMax = 1.2;

    // Control Flags (volatile for thread safety)
    private volatile boolean renderRequested = true;
    private volatile boolean shutdownRequested = false;

    // GUI and MPI Fields
    private BufferedImage image;
    private final int rank;
    private final int size;

    // Command-line flags to run diff modes
    private static boolean headlessMode = false;
    private static boolean runTests = false;

    public MandelbrotViewer(int rank, int size) {
        this.rank = rank;
        this.size = size;
        if (rank == 0) {
            this.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            if (!headlessMode && !runTests) {
                initGUI();
            }
        }
    }

    private void initGUI() {
        setTitle("Mandelbrot MPI Viewer");
        setSize(width, height);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setLocationRelativeTo(null);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                requestShutdown();
            }
        });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent ev) {
                if (renderingIsInProgress()) return;

                double panSpeed = 0.1 * (xMax - xMin);
                switch (ev.getKeyCode()) {
                    case KeyEvent.VK_LEFT:  xMin -= panSpeed; xMax -= panSpeed; break;
                    case KeyEvent.VK_RIGHT: xMin += panSpeed; xMax += panSpeed; break;
                    case KeyEvent.VK_UP:    yMin -= panSpeed; yMax -= panSpeed; break;
                    case KeyEvent.VK_DOWN:  yMin += panSpeed; yMax += panSpeed; break;
                    case KeyEvent.VK_1:     zoom(0.8); break;   // zoom out
                    case KeyEvent.VK_2:     zoom(1.25); break;  // zoom in
                    case KeyEvent.VK_S:     saveImage(); return;       // save image (S)
                    case KeyEvent.VK_Q:     requestShutdown(); return; // quit (Q)
                }
                requestRender();
            }
        });

        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                if (renderingIsInProgress()) return;
                width = getWidth();
                height = getHeight();
                if (width > 0 && height > 0) {
                    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                    requestRender();
                }
            }
        });
    }

    private void zoom(double factor) {
        double xCenter = (xMin + xMax) / 2;
        double yCenter = (yMin + yMax) / 2;
        double xRange = (xMax - xMin) * factor;
        double yRange = (yMax - yMin) * factor;
        xMin = xCenter - xRange / 2; xMax = xCenter + xRange / 2;
        yMin = yCenter - yRange / 2; yMax = yCenter + yRange / 2;
    }

    public synchronized void requestRender() { this.renderRequested = true; }
    public synchronized void requestShutdown() { this.shutdownRequested = true; }
    private synchronized boolean isRenderRequested() { return this.renderRequested; }
    private synchronized boolean isShutdownRequested() { return this.shutdownRequested; }
    private synchronized boolean renderingIsInProgress() { return this.renderRequested; }
    private synchronized void setRenderDone() { this.renderRequested = false; }

    public void runMasterLoop() {
        while (!isShutdownRequested()) {
            if (isRenderRequested()) {
                try {
                    int[] allPixels = performDistributedRender();

                    if (!headlessMode && !runTests) {
                        SwingUtilities.invokeLater(() -> {
                            image.setRGB(0, 0, width, height, allPixels, 0, width);
                            repaint();
                            setRenderDone();
                        });
                    } else {
                        setRenderDone();
                    }

                } catch (Exception e) {
                    e.printStackTrace();
                    setRenderDone();
                }
            }

        }

        try {
            System.out.println("Shutting down...");
            double[] quitSignal = new double[]{-1};
            MPI.COMM_WORLD.Bcast(quitSignal, 0, 1, MPI.DOUBLE, 0);
        } catch(MPIException e) {
            e.printStackTrace();
        }

        if (!headlessMode && !runTests) {
            dispose();
        }
    }

    private int[] performDistributedRender() throws MPIException {
        System.out.println("Rendering " + width + "x" + height + " image...");

        double[] params = new double[]{width, height, xMin, xMax, yMin, yMax};
        MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);

        int rowsPerProc = height / size;
        int extraRows = height % size;
        int yStart = rank * rowsPerProc + Math.min(rank, extraRows);
        int yEnd = yStart + rowsPerProc + (rank < extraRows ? 1 : 0);
        int[] localPixels = computeChunk(width, height, xMin, xMax, yMin, yMax, yStart, yEnd);

        int[] recvCounts = new int[size];
        int[] displs = new int[size];
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rpp = height / size;
            int er = height % size;
            int startRow = i * rpp + Math.min(i, er);
            int endRow = startRow + rpp + (i < er ? 1 : 0);
            recvCounts[i] = (endRow - startRow) * width;
            displs[i] = offset;
            offset += recvCounts[i];
        }

        int[] allPixels = new int[width * height];
        MPI.COMM_WORLD.Gatherv(localPixels, 0, localPixels.length, MPI.INT, allPixels, 0, recvCounts, displs, MPI.INT, 0);

        return allPixels;
    }

    public void runWorkerLoop() {
        while (true) {
            try {
                double[] params = new double[6];
                MPI.COMM_WORLD.Bcast(params, 0, 6, MPI.DOUBLE, 0);

                if (params[0] < 0) {
                    System.out.println("Worker " + rank + " received shutdown. Exiting.");
                    break;
                }

                int w = (int) params[0], h = (int) params[1];
                double xm = params[2], xM = params[3], ym = params[4], yM = params[5];

                int rowsPerProc = h / size;
                int extraRows = h % size;
                int yStart = rank * rowsPerProc + Math.min(rank, extraRows);
                int yEnd = yStart + rowsPerProc + (rank < extraRows ? 1 : 0);

                int[] localPixels = computeChunk(w, h, xm, xM, ym, yM, yStart, yEnd);
                MPI.COMM_WORLD.Gatherv(localPixels, 0, localPixels.length, MPI.INT, null, 0, null, null, MPI.INT, 0);
            } catch (MPIException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    private void saveImage() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image");
        fileChooser.setSelectedFile(new File("mandelbrot.png"));
        int userSelection = fileChooser.showSaveDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            new Thread(() -> {
                try {
                    ImageIO.write(image, "png", fileToSave);
                    System.out.println("Image saved to " + fileToSave.getAbsolutePath());
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(this, "Error saving file: " + ex.getMessage(), "Save Error", JOptionPane.ERROR_MESSAGE);
                }
            }).start();
        }
    }


    private void runPerformanceTests() {
        int startSize = 1000;
        int maxSize = 10000;
        String csvFile = "mandelbrot_mpi_results.csv";

        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            writer.println("width,height,distributed");

            for (int size = startSize; size <= maxSize; size += 1000) {
                // viewer's dimensions for the test
                this.width = size;
                this.height = size;

                // Start timer
                long startTime = System.currentTimeMillis();

                // Run render operation
                performDistributedRender();

                // Stop timer
                long endTime = System.currentTimeMillis();
                long renderTime = endTime - startTime;

                // Log console & write to file
                System.out.println("STATUS: Test render " + size + "x" + size + " completed in " + renderTime + " ms");
                writer.println(size + "," + size + "," + renderTime);
            }
            System.out.println("SUCCESS: Performance tests complete. Results saved to " + csvFile);
        } catch (IOException | MPIException e) {
            e.printStackTrace();
            System.err.println("ERROR: Could not write to CSV file or MPI error during test.");
        }
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        if (image != null) {
            g.drawImage(image, 0, 0, getWidth(), getHeight(), null);
        }
    }

    private static int[] computeChunk(int w, int h, double xm, double xM, double ym, double yM, int yStart, int yEnd) {
        if (yStart >= yEnd || w <= 0) return new int[0];
        int[] slicePixels = new int[(yEnd - yStart) * w];
        int idx = 0;
        for (int y = yStart; y < yEnd; y++) {
            for (int x = 0; x < w; x++) {
                double real = xm + x * (xM - xm) / (w - 1);
                double imag = ym + y * (yM - ym) / (h - 1);
                Complex c = new Complex(real, imag);
                Complex z = new Complex(0, 0);
                int n = 0;
                while (z.absSq() <= 4.0 && n < MAX_ITER) {
                    z = z.multiply(z).add(c);
                    n++;
                }
                if (n == MAX_ITER) {
                    slicePixels[idx++] = Color.BLACK.getRGB();
                } else {
                    slicePixels[idx++] = Color.getHSBColor(0.7f + (float) n / MAX_ITER, 1.0f, 1.0f).getRGB();
                }
            }
        }
        return slicePixels;
    }

    public static void main(String[] args) {
        int rank = -1;

        try {
            MPI.Init(args);
            rank = MPI.COMM_WORLD.Rank();
            int size = MPI.COMM_WORLD.Size();

            for (String arg : args) {
                if (arg.equalsIgnoreCase("--nongui")) headlessMode = true;
                else if (arg.equalsIgnoreCase("--test")) runTests = true;
            }

            MandelbrotViewer viewer = new MandelbrotViewer(rank, size);

            if (rank == 0) {
                if (headlessMode) {
                    System.out.println("Running in headless (non-GUI) mode.");
                    long startTime = System.currentTimeMillis();
                    viewer.performDistributedRender();
                    long endTime = System.currentTimeMillis();
                    long renderTime = endTime - startTime;
                    System.out.println("Headless render complete in: " + renderTime + " ms");
                    viewer.requestShutdown();
                    viewer.runMasterLoop();
                } else if (runTests) {
                    System.out.println("Running performance tests with " + size + " processes...");
                    viewer.runPerformanceTests(); // Call the test runner
                    viewer.requestShutdown();
                    viewer.runMasterLoop(); // Run loop to send shutdown signal
                } else {
                    SwingUtilities.invokeLater(() -> viewer.setVisible(true));
                    viewer.runMasterLoop();
                }
            } else {
                viewer.runWorkerLoop();
            }
        } catch (MPIException e) {
            e.printStackTrace();
        } finally {
            try {
                MPI.Finalize();
                if(rank != -1) { System.out.println("Process " + rank + " finalized."); }
            } catch (MPIException e) { e.printStackTrace(); }
        }
    }

    static class Complex {
        private final double real, imag;
        public Complex(double r, double i) { real = r; imag = i; }
        public Complex add(Complex o) { return new Complex(real + o.real, imag + o.imag); }
        public Complex multiply(Complex o) { return new Complex(real * o.real - imag * o.imag, real * o.imag + imag * o.real); }
        public double absSq() { return real * real + imag * imag; }
    }
}