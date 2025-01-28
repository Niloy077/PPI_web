const app = Vue.createApp({
    data() {
        return {
            pdbFile: null,             // Stores the uploaded PDB file
            pdbFileName: '',           // Name of the uploaded file
            fastaSequence: '',         // User-input FASTA sequence
            isFastaValid: false        // Validation state for FASTA sequence
        };
    },
    methods: {
        // Handle PDB file upload
        handlePdbFile(event) {
            const file = event.target.files[0];
            if (file && file.name.endsWith('.pdb')) {
                this.pdbFile = file;
                this.pdbFileName = file.name;
                document.getElementById('pdb-submit').disabled = false;
                document.getElementById('pdb-uploaded').textContent = `Uploaded: ${file.name}`;
            } else {
                alert('Please upload a valid PDB file.');
            }
        },
        // Validate FASTA sequence
        validateFasta(event) {
            const fastaRegex = /^[ACDEFGHIKLMNPQRSTVWY\s]+$/i; // Standard FASTA characters
            const textarea = event.target;
            this.isFastaValid = fastaRegex.test(textarea.value);
            document.getElementById('fasta-submit').disabled = !this.isFastaValid;
            document.getElementById('fasta-validation').style.display = this.isFastaValid ? 'none' : 'block';
        },
        // Submit PDB file
        submitPdbFile() {
            alert(`PDB file "${this.pdbFileName}" submitted successfully!`);
        },
        // Submit FASTA sequence
        submitFasta() {
            alert('FASTA sequence submitted successfully!');
        }
    },
    mounted() {
        document.getElementById('pdb-input').addEventListener('change', this.handlePdbFile);
        document.getElementById('fasta-input').addEventListener('input', this.validateFasta);
        document.getElementById('pdb-submit').addEventListener('click', this.submitPdbFile);
        document.getElementById('fasta-submit').addEventListener('click', this.submitFasta);
    }
});

app.mount('#app');
