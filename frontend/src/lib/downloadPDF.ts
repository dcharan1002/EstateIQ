import html2canvas from 'html2canvas-pro';
import jsPDF from 'jspdf';

export async function downloadPDF(element: HTMLElement, filename: string = 'report.pdf', isDarkMode: boolean = false) {
  const canvas = await html2canvas(element, {
    scale: 2,
    logging: false,
    useCORS: true,
    backgroundColor: isDarkMode ? '#1f2937' : '#ffffff',
    windowWidth: 1024, // Fixed width for consistent rendering
    width: 800 // Control content width
  });
  
  const imgData = canvas.toDataURL('image/png');
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });
  
  const imgWidth = 190; // A4 width minus margins
  const pageHeight = 277; // A4 height minus margins
  const imgHeight = (canvas.height * imgWidth) / canvas.width;
  let heightLeft = imgHeight;
  let position = 10; // Top margin

  pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight); // Add left margin
  heightLeft -= pageHeight;

  while (heightLeft >= pageHeight) {
    position = heightLeft - imgHeight + 10; // Add margin to subsequent pages
    pdf.addPage();
    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
    heightLeft -= pageHeight - 20; // Account for top and bottom margins
  }

  pdf.save(filename);
}
