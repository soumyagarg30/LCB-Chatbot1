import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { adminLogin, googleLogin } from "@/utils/api";

const Login = () => {
  const navigate = useNavigate();
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const existingToken = localStorage.getItem("lcb_auth_token");
    if (existingToken) {
      navigate("/", { replace: true });
    }
  }, [navigate]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);

    try {
      if (isSignup) {
        // Signup
        const response = await fetch("http://localhost:5001/api/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, name }),
        });

        const data = await response.json();
        if (response.ok && data.token) {
          localStorage.setItem("lcb_auth_token", data.token);
          localStorage.setItem("lcb_user", JSON.stringify(data.user));
          toast.success(data.message || "Account created successfully!");
          navigate("/", { replace: true });
        } else {
          toast.error(data.error || "Signup failed");
        }
      } else {
        // Login
        const response = await fetch("http://localhost:5001/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        const data = await response.json();
        if (response.ok && data.token) {
          localStorage.setItem("lcb_auth_token", data.token);
          localStorage.setItem("lcb_user", JSON.stringify(data.user));
          toast.success(data.message || "Signed in successfully!");
          navigate("/", { replace: true });
        } else {
          toast.error(data.error || "Login failed");
        }
      }
    } catch (error) {
      console.error(error);
      toast.error(isSignup ? "Signup failed" : "Login failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 px-4 py-10 text-white">
      <div className="mx-auto flex max-w-5xl flex-col gap-8 lg:flex-row lg:items-center">
        <div className="flex-1 space-y-5">
          <div className="inline-flex items-center rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-sm text-emerald-200">
            Secure access to LCB Assistant
          </div>
          <div className="space-y-3">
            <h1 className="text-4xl font-semibold sm:text-5xl">
              {isSignup ? "Create your account" : "Welcome back"}
            </h1>
            <p className="max-w-xl text-lg text-slate-300">
              {isSignup
                ? "Sign up to access the LCB AI Assistant. Join our community of agricultural professionals."
                : "Log in to continue to the LCB AI Assistant and get instant answers to your farming questions."}
            </p>
          </div>
        </div>

        <div className="w-full max-w-md rounded-[2rem] border border-white/10 bg-slate-900/80 p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignup && (
              <div>
                <label className="mb-2 block text-sm text-slate-300">Full name</label>
                <Input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Your name"
                  className="rounded-2xl border border-white/10 bg-slate-950/70 text-white"
                  required
                />
              </div>
            )}

            <div>
              <label className="mb-2 block text-sm text-slate-300">Email</label>
              <Input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder={isSignup ? "you@example.com" : "admin@lcb.com"}
                className="rounded-2xl border border-white/10 bg-slate-950/70 text-white"
                required
              />
            </div>

            <div>
              <label className="mb-2 block text-sm text-slate-300">Password</label>
              <Input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder={isSignup ? "At least 6 characters" : "LCB@1234"}
                className="rounded-2xl border border-white/10 bg-slate-950/70 text-white"
                required
              />
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full rounded-full bg-emerald-500 py-3 text-white hover:bg-emerald-400"
            >
              {isLoading ? (isSignup ? "Creating account..." : "Signing in...") : isSignup ? "Create account" : "Sign in"}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-slate-400">
              {isSignup ? "Already have an account?" : "Don't have an account?"}{" "}
              <button
                type="button"
                onClick={() => {
                  setIsSignup(!isSignup);
                  setEmail("");
                  setPassword("");
                  setName("");
                }}
                className="font-semibold text-emerald-400 hover:text-emerald-300"
              >
                {isSignup ? "Sign in" : "Sign up"}
              </button>
            </p>
          </div>

          {!isSignup && (
            <div className="mt-4 rounded-lg border border-white/10 bg-white/5 p-3 text-xs text-slate-300">
              <p className="font-semibold text-emerald-200">Demo Admin Account:</p>
              <p>Email: admin@lcb.com</p>
              <p>Password: LCB@1234</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Login;

