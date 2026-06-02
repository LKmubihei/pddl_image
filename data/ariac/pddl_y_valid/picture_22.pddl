(define (problem picture_22)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_pump blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_pump)
        (clear red_pump)
        (clear blue_pump)
        (part_at blue_pump table)
        (part_at green_pump table)
        (part_at red_pump table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
