// ====================== TOGGLE ANIMATION =======================
const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

sign_up_btn.addEventListener("click", () => {
    container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener("click", () => {
    container.classList.remove("sign-up-mode");
});


// ====================== SIGNUP =======================
async function signupUser() {
    let full_name = document.getElementById("signup_username").value.trim();
    let email = document.getElementById("signup_email").value.trim();
    let password = document.getElementById("signup_password").value.trim();

    if (!full_name || !email || !password) {
        alert("Please fill all fields!");
        return;
    }

    try {
        const res = await fetch("/signup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ full_name, email, password })
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || "Signup failed");
            return;
        }

        alert("Signup successful! Please login.");
        container.classList.remove("sign-up-mode");

    } catch (err) {
        alert("Signup error: " + err.message);
    }
}



// ====================== LOGIN =======================
async function loginUser() {
    let username = document.getElementById("login_username").value.trim();
    let password = document.getElementById("login_password").value.trim();

    if (!username || !password) {
        alert("Please enter both email and password");
        return;
    }

    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    try {
        const res = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || "Invalid login");
            return;
        }

        const data = await res.json();
        console.log("Login OK:", data);

        // FIXED → Store token with correct key
        localStorage.setItem("token", data.access_token);

        // Redirect
        window.location.href = "/dashboard";

    } catch (err) {
        alert("Login error: " + err.message);
    }
}



// ====================== BIND FORM SUBMIT =======================
document.getElementById("loginForm").addEventListener("submit", function (e) {
    e.preventDefault();
    loginUser();
});

document.getElementById("signupForm").addEventListener("submit", function (e) {
    e.preventDefault();
    signupUser();
});
