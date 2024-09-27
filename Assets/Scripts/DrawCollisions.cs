using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteAlways]
public class DrawCollisions : MonoBehaviour
{
    private void OnDrawGizmos()
    {
        Collider collider = GetComponent<Collider>();

        if (collider != null)
        {
            Gizmos.color = Color.green; // Set the color for the gizmo

            // Draw based on the collider type
            if (collider is BoxCollider box)
            {
                Gizmos.matrix = transform.localToWorldMatrix; // Ensure it draws in the correct orientation
                Gizmos.DrawWireCube(box.center, box.size);
            }
            else if (collider is SphereCollider sphere)
            {
                Gizmos.DrawWireSphere(transform.position + sphere.center, sphere.radius);
            }
            else if (collider is CapsuleCollider capsule)
            {
                float radius = capsule.radius;
                float height = capsule.height / 2 - radius;

                Vector3 capsuleCenter = transform.position + capsule.center;

                Gizmos.DrawWireSphere(capsuleCenter + transform.up * height, radius); // Top sphere
                Gizmos.DrawWireSphere(capsuleCenter - transform.up * height, radius); // Bottom sphere
                Gizmos.DrawLine(capsuleCenter + transform.up * height + transform.right * radius,
                                capsuleCenter - transform.up * height + transform.right * radius);
                Gizmos.DrawLine(capsuleCenter + transform.up * height - transform.right * radius,
                                capsuleCenter - transform.up * height - transform.right * radius);
                Gizmos.DrawLine(capsuleCenter + transform.up * height + transform.forward * radius,
                                capsuleCenter - transform.up * height + transform.forward * radius);
                Gizmos.DrawLine(capsuleCenter + transform.up * height - transform.forward * radius,
                                capsuleCenter - transform.up * height - transform.forward * radius);
            }
        }
    }
}
